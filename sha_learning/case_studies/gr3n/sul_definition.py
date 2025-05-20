import configparser
import os
from typing import List
import matplotlib.pyplot as plt

from sha_learning.case_studies.gr3n.sul_functions import label_event, parse_data, get_absorption_param, is_chg_pt
from sha_learning.case_studies.gr3n.sul_functions import plot_assorbimento_eventi, plot_coppia_eventi
from sha_learning.domain.lshafeatures import Event, NormalDistribution, Trace
from sha_learning.domain.sigfeatures import Timestamp, SampledSignal
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from sha_learning.learning_setup.teacher import Teacher
from sha_learning.pltr.gr3n_pltr import distr_hist

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

COPPIA_RANGE = int(config['GR3N']['COPPIA_RANGE'])
MIN_COPPIA = int(config['GR3N']['MIN_COPPIA'])
MAX_COPPIA = int(config['GR3N']['MAX_COPPIA'])

DIF_RANGE = int(config['GR3N']['DIF_RANGE'])
MIN_DIF = int(config['GR3N']['MIN_DIF'])
MAX_DIF = int(config['GR3N']['MAX_DIF'])


def modello_assorbimento(interval: List[Timestamp], P_0):
    interval = [ts.to_secs() for ts in interval]
    AVG_PW = 1.0
    return [AVG_PW] * len(interval)


# define flow conditions
on_fc: FlowCondition = FlowCondition(0, modello_assorbimento)

# define distributions
off_distr = NormalDistribution(0, 0.0, 0.0)

model2distr = {0: []}


assorbimento = RealValuedVar([on_fc], [], model2distr, label='a')

# define events
events: List[Event] = []
for i in range(MIN_COPPIA, MAX_COPPIA, COPPIA_RANGE):
    if i < MAX_COPPIA - COPPIA_RANGE:
        new_guard = '{}<=cp<{}'.format(i, i + COPPIA_RANGE)
    else:
        new_guard = '{}<=cp'.format(i)
    events.append(Event(new_guard, 'start', 'cp_{}'.format(len(events))))

for i in range(MIN_DIF, MAX_DIF, DIF_RANGE):
    if i < MAX_DIF - DIF_RANGE:
        new_guard = '{}<=cp<{}'.format(i, i + DIF_RANGE)
    else:
        new_guard = '{}<=cp'.format(i)
    events.append(Event(new_guard, 'start', 'cp_{}'.format(len(events))))

events.append(Event('', 'stop', 's'))

DRIVER_SIG = ['cp', 'df']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'assorbimento', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}

gr3n_cs = SystemUnderLearning([assorbimento], events, parse_data, label_event, get_absorption_param, is_chg_pt, args=args)

test = False
if test:
    TEACHER = Teacher(gr3n_cs)

    TEST_PATH = config['GR3N']['CV_PATH']
    N = 10
    traces_files = os.listdir(TEST_PATH)
    traces_files = [file for file in traces_files]
    traces_files.sort()
    for file in traces_files:
        # testing data to signals conversion
        new_signals: List[SampledSignal] = parse_data(TEST_PATH + file)
        '''
        sufficies to put here an if that states that if there are not enough data you skip,
        but the problem still remains during the learning, in that case, I think I should search for 
        the correct files that contains the kind of data that I want (coppia and assorbimento in this case)
        '''
        # testing chg pts identification
        chg_pts = gr3n_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
        # testing event labeling
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]
        # testing signal to trace conversion
        gr3n_cs.process_data(TEST_PATH + file)
        trace = gr3n_cs.timed_traces[-1]
        print('{}\t{}\t{}\t{}'.format(file, Trace(tt=trace),
                                      trace.t[-1].to_secs() - trace.t[0].to_secs(), len(trace)))
        #plot_coppia_eventi(TEST_PATH + file, file, trace)
        for j, event in enumerate(trace.e):
            test_trace = Trace(gr3n_cs.traces[-1][:j])
            identified_model: FlowCondition = TEACHER.mi_query(test_trace)
            print(identified_model)
            identified_distr = TEACHER.ht_query(test_trace, identified_model, save=True)
            segments = gr3n_cs.get_segments(test_trace)
            avg_metrics = sum([TEACHER.sul.get_ht_params(segment, identified_model)
                               for segment in segments]) / len(segments)

            try:
                print('{}:\t{:.3f}->{}'.format(test_trace.events[-1].symbol, avg_metrics, identified_distr.params))
            except IndexError:
                print('{}:\t{:.3f}->{}'.format(test_trace, avg_metrics, identified_distr.params))

    # test segment identification
    test_trace = Trace(gr3n_cs.traces[0][:1])
    segments = gr3n_cs.get_segments(test_trace)

    # test model identification

    # test distr identification
    for i, trace in enumerate(TEACHER.timed_traces):
        for j, event in enumerate(trace.e):
            test_trace = Trace(gr3n_cs.traces[i][:j])
            identified_distr = TEACHER.ht_query(test_trace, identified_model, save=True)
            segments = gr3n_cs.get_segments(test_trace)
            avg_metrics = sum([TEACHER.sul.get_ht_params(segment, identified_model)
                               for segment in segments]) / len(segments)

            try:
                print('{}:\t{:.3f}->{}'.format(test_trace.events[-1].symbol, avg_metrics, identified_distr.params))
            except IndexError:
                print('{}:\t{:.3f}->{}'.format(test_trace, avg_metrics, identified_distr.params))

    for d in gr3n_cs.vars[0].distr:
        print(d.params)
    distr_hist(TEACHER.hist, 'gr3n')

