import configparser
import os
from typing import List

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
events.append(Event('', 'cp high', 'cp1'))
events.append(Event('', 'cp low', 'cp2'))

DRIVER_SIG = ['cp']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'assorbimento', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}

gr3n_cs = SystemUnderLearning([assorbimento], events, parse_data, label_event, get_absorption_param, is_chg_pt, args=args)

test = True
if test:
    TEST_PATH = config['GR3N']['CV_PATH']
    N = 10
    traces_files = os.listdir(TEST_PATH)
    traces_files = [file for file in traces_files]
    traces_files.sort()
    for file in traces_files:
        # testing data to signals conversion
        new_signals: List[SampledSignal] = parse_data(TEST_PATH + file)
        # testing chg pts identification
        chg_pts = gr3n_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
        # testing event labeling
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts[:10]]
        # testing signal to trace conversion
        gr3n_cs.process_data(TEST_PATH + file)
        trace = gr3n_cs.timed_traces[-1]
        print('{}\t{}\t{}\t{}'.format(file, Trace(tt=trace),
                                      trace.t[-1].to_secs() - trace.t[0].to_secs(), len(trace)))

        plot_assorbimento_eventi(trace)
        plot_coppia_eventi(trace)

    # test segment identification
    test_trace = Trace(gr3n_cs.traces[0][:1])
    segments = gr3n_cs.get_segments(test_trace)

    # test model identification
    TEACHER = Teacher(gr3n_cs)
    identified_model: FlowCondition = TEACHER.mi_query(test_trace)
    print(identified_model)

    # test distr identification
    for i, trace in enumerate(TEACHER.timed_traces):
        for j, event in enumerate(trace.e):
            test_trace = Trace(gr3n_cs.traces[i][:j])
            identified_distr = TEACHER.ht_query(test_trace, identified_model, save=True)
            '''
         Qui si genera errore perchè crea dei segmenti di dati (credo sulla base degli eventi riconosciuti)
            e ad un certo punto crea un segmento nullo (come è possibile?) e va a calcolare una roba che
            così non va bene.
            Questo succede perchè si generano due timestamp attaccati uno all'altro (anzi, uno sembra quasi uscito 
            dal nulla, perchè nella time_trace del sul non c'è... indici di array 260 e 261, o giù di li) e quindi 
            non trova niente nel mezzo
            '''
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

