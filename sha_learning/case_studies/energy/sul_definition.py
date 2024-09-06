import configparser
import os
from typing import List, Set, Tuple

from pygad import GA

from sha_learning.case_studies.energy.sul_functions import label_event, parse_data, get_power_param, is_chg_pt
from sha_learning.domain.lshafeatures import Event, NormalDistribution, Trace
from sha_learning.domain.sigfeatures import Timestamp, SampledSignal
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])


def pwr_model(interval: List[Timestamp], P_0):
    interval = [ts.to_secs() for ts in interval]
    AVG_PW = 1.0
    return [AVG_PW] * len(interval)


# define flow conditions
on_fc: FlowCondition = FlowCondition(0, pwr_model)

# define distributions
off_distr = NormalDistribution(0, 0.0, 0.0)

model2distr = {0: []}
power = RealValuedVar([on_fc], [], model2distr, label='P')

# define events
events: List[Event] = []
for i in range(MIN_SPEED, MAX_SPEED, SPEED_RANGE):
    if i < MAX_SPEED - SPEED_RANGE:
        new_guard = '{}<=w<{}'.format(i, i + SPEED_RANGE)
    else:
        new_guard = '{}<=w'.format(i)
    events.append(Event(new_guard, 'start', 'm_{}'.format(len(events))))

spindle_off = Event('', 'stop', 'i_0')

events.append(spindle_off)
events.append(Event('', 'load', 'l'))
events.append(Event('', 'unload', 'u'))

DRIVER_SIG = ['w', 'pr']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'energy', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
energy_cs = SystemUnderLearning([power], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)

test = False
if test:
    TEST_PATH = config['TRACE GENERATION']['SIM_LOGS_PATH'].format('ENERGY')
    N = 10
    traces_files = os.listdir(TEST_PATH)
    traces_files = [file for file in traces_files if file[0] in ['_', 'W']]
    traces_files.sort()
    for file in traces_files:
        # testing data to signals conversion
        new_signals: List[SampledSignal] = parse_data(TEST_PATH + file)
        # testing chg pts identification
        chg_pts = energy_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
        # testing event labeling
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]
        # testing signal to trace conversion
        energy_cs.process_data(TEST_PATH + file)
        trace = energy_cs.timed_traces[-1]
        print('{}\t{}\t{}\t{}'.format(file, Trace(tt=trace),
                                      trace.t[-1].to_secs() - trace.t[0].to_secs(), len(trace)))
        # power_pts = new_signals[0].points
        # speed_pts = new_signals[1].points
        # pressure_pts = new_signals[2].points
        # sim_power = [0.0, 0.0, 8.2, 0.0, 3.12, 0.0, 2.4, 0.0, 1.5, 8.0, 0.0, 0.8, 0.0]
        # single_plot([pt.timestamp for pt in power_pts], [pt.value for pt in power_pts],
        #             energy_cs.timed_traces[-1].t, sim_power, trace)
        # double_plot([pt.timestamp for pt in power_pts], [pt.value for pt in power_pts],
        #           [pt.timestamp for pt in speed_pts], [pt.value for pt in speed_pts],
        #           trace, title=file, filtered=True,
        #           timestamps3=[pt.timestamp for pt in pressure_pts],
        #           v3=[pt.value for pt in pressure_pts])
        pass

    unique_events: List[Tuple[str, Set[Event]]] = []
    for i, t_trace in enumerate(energy_cs.timed_traces):
        trace = Trace(tt=t_trace)
        unique_events.append((traces_files[i], set(trace.events)))
    unique_events.sort(key=lambda s: len(s[1]), reverse=True)
    print(unique_events)

    unique_events.sort(key=lambda s: s[0])
    print(unique_events)


    def fitness_function(ga_instance: GA, solution, solution_idx):
        output = [unique_events[i] for i in solution]
        output_set = set()
        for tt in output:
            output_set = output_set.union(tt[1])
        return len(output_set) * sum([len(tt[1]) for tt in output])


    CLUSTER_DIM = 9

    ga_instance = GA(num_generations=50,
                     num_parents_mating=4,
                     fitness_func=fitness_function,
                     sol_per_pop=8,
                     num_genes=CLUSTER_DIM,
                     gene_type=int,
                     init_range_low=0,
                     init_range_high=len(energy_cs.timed_traces) - 1,
                     parent_selection_type="sss",
                     keep_parents=1,
                     crossover_type="single_point",
                     mutation_type="random",
                     mutation_percent_genes=10)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    # test segment identification
    # test_trace = Trace(energy_cs.traces[0][:1])
    # segments = energy_cs.get_segments(test_trace)
    #
    # # test model identification
    # TEACHER = Teacher(energy_cs)
    # identified_model: FlowCondition = TEACHER.mi_query(test_trace)
    # print(identified_model)
    #
    # # test distr identification
    # for i, trace in enumerate(TEACHER.timed_traces):
    #     for j, event in enumerate(trace.e):
    #         test_trace = Trace(energy_cs.traces[i][:j])
    #         identified_distr = TEACHER.ht_query(test_trace, identified_model, save=True)
    #
    #         segments = energy_cs.get_segments(test_trace)
    #         avg_metrics = sum([TEACHER.sul.get_ht_params(segment, identified_model)
    #                            for segment in segments]) / len(segments)
    #
    #         try:
    #             print('{}:\t{:.3f}->{}'.format(test_trace.events[-1].symbol, avg_metrics, identified_distr.params))
    #         except IndexError:
    #             print('{}:\t{:.3f}->{}'.format(test_trace, avg_metrics, identified_distr.params))
    #
    # for d in energy_cs.vars[0].distr:
    #     print(d.params)
    # distr_hist(TEACHER.hist)
