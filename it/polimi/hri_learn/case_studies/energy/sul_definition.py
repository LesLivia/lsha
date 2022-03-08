from typing import List

from it.polimi.hri_learn.case_studies.energy.sul_functions import label_event, parse_data, get_power_param
from it.polimi.hri_learn.domain.lshafeatures import Event, NormalDistribution, Trace
from it.polimi.hri_learn.domain.sigfeatures import Timestamp, SampledSignal
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from it.polimi.hri_learn.lstar_sha.teacher import Teacher


# FIXME: temporarily approximated to constant function
def pwr_model(interval: List[Timestamp], P_0):
    interval = [ts.to_secs() for ts in interval]
    AVG_PW = 1.0
    return [AVG_PW] * len(interval)


# define flow conditions
on_fc: FlowCondition = FlowCondition(0, pwr_model)

# define distributions
off_distr = NormalDistribution(0, 0.0, 0.1)

model2distr = {0: [0]}
power = RealValuedVar([on_fc], [off_distr], model2distr, label='P')

# define events
spindle_on1 = Event('100<=w<250', 'start', 'm_0')
spindle_on2 = Event('250<=w<500', 'start', 'm_1')
spindle_on3 = Event('500<=w<750', 'start', 'm_2')
spindle_on4 = Event('750<=w<1000', 'start', 'm_3')
spindle_on5 = Event('1000<=w<1250', 'start', 'm_4')
spindle_on6 = Event('1250<=w<1500', 'start', 'm_5')
spindle_on7 = Event('1500<=w<1750', 'start', 'm_6')
spindle_on8 = Event('1750<=w<2000', 'start', 'm_7')
spindle_on9 = Event('2000<=w', 'start', 'm_8')

spindle_off = Event('', 'stop', 'i_0')

events = [spindle_off, spindle_on1, spindle_on2, spindle_on3, spindle_on4, spindle_on5,
          spindle_on6, spindle_on7, spindle_on8, spindle_on9]

DRIVER_SIG = 'w'
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'energy', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
energy_cs = SystemUnderLearning([power], events, parse_data, label_event, get_power_param, args=args)

test = False
if test:
    TEST_PATH = '/Users/lestingi/PycharmProjects/lsha/resources/traces/simulations/energy/_W9_2019-10-31_6-8.csv'
    # testing data to signals conversion
    new_signals: List[SampledSignal] = parse_data(TEST_PATH)

    # testing chg pts identification
    chg_pts = SystemUnderLearning.find_chg_pts([sig for sig in new_signals if sig.label == DRIVER_SIG][0])

    # testing event labeling
    id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]

    # testing signal to trace conversion
    new_trace = energy_cs.process_data(TEST_PATH)
    for trace in energy_cs.traces:
        print(trace)

    # test segment identification
    test_trace = Trace([spindle_on6])
    segments = energy_cs.get_segments(test_trace)
    print(segments)

    # test model identification
    TEACHER = Teacher(energy_cs)
    identified_model: FlowCondition = TEACHER.mi_query(test_trace)
    print(identified_model)
    identified_distr = TEACHER.ht_query(test_trace, identified_model, save=True)
    print(identified_distr)
    for d in energy_cs.vars[0].distr:
        print(d.params)
