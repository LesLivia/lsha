import configparser
import math
import os
from typing import List

from sha_learning.case_studies.hri.sul_functions import label_event, parse_data, get_ftg_param, \
    is_chg_pt
from sha_learning.domain.lshafeatures import RealValuedVar, FlowCondition, Trace
from sha_learning.domain.sigfeatures import Event, Timestamp
from sha_learning.domain.sulfeatures import SystemUnderLearning
from sha_learning.learning_setup.teacher import Teacher
from sha_learning.pltr.hri_pltr import double_plot

N_0 = (0.003, 0.0001, 100)
N_1 = (0.004, 0.0004, 100)

MAIN_SIGNAL = 0
DRIVER_SIG = '2'
DEFAULT_M = 0
DEFAULT_DISTR = 0

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()
CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', '')[0])
SAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY']


def idle_model(interval: List[Timestamp], F_0: float):
    interval = [ts.to_secs() for ts in interval]
    return [F_0 * math.exp(-N_0[0] * (t - interval[0])) for t in interval]


def busy_model(interval: List[Timestamp], F_0: float):
    interval = [ts.to_secs() for ts in interval]
    return [1 - (1 - F_0) * math.exp(-N_1[0] * (t - interval[0])) for t in interval]


idle_fc = FlowCondition(0, idle_model)
busy_fc = FlowCondition(1, busy_model)

models: List[FlowCondition] = [idle_fc, busy_fc]

events = []
if SAMPLE_STRATEGY == 'UPPAAL':
    if CS_VERSION in [1]:
        u_event = Event('', 'start', 'u_1')
        d_event = Event('', 'stop', 'd_1')
        events = [u_event, d_event]
    elif CS_VERSION in [2]:
        u_event1 = Event('sit', 'start', 'u_1')
        d_event1 = Event('sit', 'stop', 'd_1')
        u_event2 = Event('!sit', 'start', 'u_2')
        d_event2 = Event('!sit', 'stop', 'd_2')
        events = [u_event1, d_event1, u_event2, d_event2]
    elif CS_VERSION in [3, 4, 5]:
        u_event1 = Event('sit!run', 'start', 'u_2')
        d_event1 = Event('sit!run', 'stop', 'd_2')
        u_event2 = Event('!sitrun', 'start', 'u_3')
        d_event2 = Event('!sitrun', 'stop', 'd_3')
        u_event3 = Event('!sit!run', 'start', 'u_4')
        d_event3 = Event('!sit!run', 'stop', 'd_4')
        events = [u_event1, d_event1, u_event2, d_event2, u_event3, d_event3]
else:
    if CS_VERSION in [1, 2, 3]:
        u_event1 = Event('sit!run', 'start', 'u_2')
        d_event1 = Event('sit!run', 'stop', 'd_2')
        u_event2 = Event('!sitrun', 'start', 'u_3')
        d_event2 = Event('!sitrun', 'stop', 'd_3')
        u_event3 = Event('!sit!run', 'start', 'u_4')
        d_event3 = Event('!sit!run', 'stop', 'd_4')
        events = [u_event1, d_event1, u_event2, d_event2, u_event3, d_event3]
    elif CS_VERSION in [4]:
        u_j = Event('!srh!l!a', 'start', 'u_j')
        u_n = Event('!sr!h!l!a', 'start', 'u_n')
        u_p = Event('!s!rhl!a', 'start', 'u_p')
        u_r = Event('!s!rh!l!a', 'start', 'u_r')
        u_t = Event('!s!r!hl!a', 'start', 'u_t')
        u_u = Event('!s!r!h!la', 'start', 'u_u')
        u_v = Event('!s!r!h!l!a', 'start', 'u_v')
        d_7 = Event('sr!h!l!a', 'stop', 'd_7')
        d_b = Event('s!rh!l!a', 'stop', 'd_b')
        d_f = Event('s!r!h!l!a', 'stop', 'd_f')
        d_j = Event('!srh!l!a', 'stop', 'd_j')
        d_p = Event('!s!rhl!a', 'stop', 'd_p')
        d_r = Event('!s!rh!l!a', 'stop', 'd_r')
        d_v = Event('!s!r!h!l!a', 'stop', 'd_v')
        events = [u_j, u_n, u_p, u_r, u_t, u_u, u_v, d_7, d_b, d_f, d_j, d_p, d_r, d_v]

model_to_distr = {}
for m in models:
    model_to_distr[m.f_id] = []

fatigue = RealValuedVar(models, [], model_to_distr, label='0')
args = {'name': 'hri', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
hri_cs = SystemUnderLearning([fatigue], events, parse_data, label_event, get_ftg_param, is_chg_pt,
                             args=args)

test = False
if test:
    # test event configuration
    print(hri_cs.symbols)

    # test trace processing
    CS_VERSION = config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', '')
    TRACE_PATH = config['TRACE GENERATION']['SIM_LOGS_PATH'].format(os.environ['LSHA_RES_PATH'], CS_VERSION)
    hri_traces = [file for file in os.listdir(TRACE_PATH) if file.startswith('SIM')]
    hri_traces.sort()
    for trace in hri_traces:
        hri_cs.process_data(TRACE_PATH + '/' + trace + '/')
        print(hri_cs.traces[-1])

    test_trace = Trace([events[2], events[3]])
    plot_traces = [(i, t) for i, t in enumerate(hri_cs.traces) if t.startswith(test_trace)]

    # test segment identification
    segments = hri_cs.get_segments(test_trace)
    print(len(segments))

    # test model identification query
    teacher = Teacher(hri_cs)
    id_flow = teacher.mi_query(test_trace)
    print(id_flow)

    # test hypothesis testing query
    metrics = [get_ftg_param(s, id_flow) for s in segments]
    print(metrics)
    print(hri_cs.vars[0].model2distr[1])
    print(teacher.ht_query(test_trace, id_flow, save=True))

    for tup in plot_traces:
        double_plot(teacher.signals[tup[0]][0], teacher.signals[tup[0]][1], teacher.signals[tup[0]][3],
                    teacher.sul.timed_traces[tup[0]], str(tup[1]), events)
