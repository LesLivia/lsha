import configparser
import os
from typing import List

from it.polimi.hri_learn.case_studies.energy_made.sul_functions import label_event, parse_data, get_power_param, \
    is_chg_pt
from it.polimi.hri_learn.domain.lshafeatures import Event, NormalDistribution, Trace
from it.polimi.hri_learn.domain.sigfeatures import Timestamp, SampledSignal
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from it.polimi.hri_learn.lstar_sha.teacher import Teacher
from it.polimi.hri_learn.pltr.energy_made_pltr import double_plot

config = configparser.ConfigParser()  # open the configuration file
config.sections()
config.read('./resources/config/config.ini')
config.sections()

SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])  # get all the constant values from the config file
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

# define events as different velocities ranges and stop, load and unload
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

DRIVER_SIG = ['w', 'pr', 'id', 'wd']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'energy', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
energy_made_cs = SystemUnderLearning([power], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)

test = False
if test:
    TEST_PATH = config['TRACE GENERATION']['SIM_LOGS_PATH'].replace('{}/', 'ENERGY/')
    traces_files = os.listdir(TEST_PATH)
    traces_files = [f for f in traces_files if f.startswith('_')]
    traces_files.sort()

    for file in traces_files:
        # testing data to signals conversion
        new_signals: List[SampledSignal] = parse_data(TEST_PATH + file)
        # testing chg pts identification
        chg_pts = energy_made_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
        # testing event labeling
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]
        # testing signal to trace conversion
        energy_made_cs.process_data(TEST_PATH + file)
        trace = energy_made_cs.timed_traces[-1]
        print(Trace(tt=trace))
        power_pts = new_signals[0].points
        speed_pts = new_signals[1].points
        pressure_pts = new_signals[2].points
        double_plot([pt.timestamp for pt in power_pts], [pt.value for pt in power_pts],
                    [pt.timestamp for pt in speed_pts], [pt.value for pt in speed_pts],
                    trace, title=file, filtered=True,
                    timestamps3=[pt.timestamp for pt in pressure_pts],
                    v3=[pt.value for pt in pressure_pts])

    # test segment identification
    test_trace = Trace(energy_made_cs.traces[0][:1])
    segments = energy_made_cs.get_segments(test_trace)

    # test model identification
    TEACHER = Teacher(energy_made_cs)
    identified_model: FlowCondition = TEACHER.mi_query(test_trace)
    print(identified_model)

    # test distr identification
    for i, trace in enumerate(TEACHER.timed_traces):
        for j, event in enumerate(trace.e):
            test_trace = Trace(energy_made_cs.traces[i][:j + 1])
            identified_distr = TEACHER.ht_query(test_trace, identified_model, save=True)

            segments = energy_made_cs.get_segments(test_trace)
            avg_metrics = sum([TEACHER.sul.get_ht_params(segment, identified_model)
                               for segment in segments]) / len(segments)

            try:
                print('{}:\t{:.3f}->{}'.format(test_trace.events[-1].symbol, avg_metrics, identified_distr.params))
            except IndexError:
                print('{}:\t{:.3f}->{}'.format(test_trace, avg_metrics, identified_distr.params))