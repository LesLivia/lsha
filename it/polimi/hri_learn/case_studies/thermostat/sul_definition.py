import configparser
import math
import os
from typing import List

from it.polimi.hri_learn.case_studies.thermostat.sul_functions import label_event, parse_data, get_thermo_param, \
    is_chg_pt
from it.polimi.hri_learn.domain.lshafeatures import RealValuedVar, FlowCondition, Trace
from it.polimi.hri_learn.domain.sigfeatures import Event, Timestamp
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning
from it.polimi.hri_learn.lstar_sha.teacher import Teacher

CLOSED_R = 100.0
OFF_DISTR = (100.0, 1.0, 200)
ON_DISTR = (0.7, 0.01, 200)
DRIVER_SIG = 't.ON'
DEFAULT_M = 1
DEFAULT_DISTR = 1

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()
CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))


def off_model(interval: List[Timestamp], T_0: float):
    interval = [ts.to_secs() for ts in interval]
    return [T_0 * math.exp(-1 / OFF_DISTR[0] * (t - interval[0])) for t in interval]


def on_model(interval: List[Timestamp], T_0: float):
    interval = [ts.to_secs() for ts in interval]
    coeff = CLOSED_R * ON_DISTR[0]
    return [coeff - (coeff - T_0) * math.exp(-(1 / CLOSED_R) * (t - interval[0])) for t in interval]


def off_model_2(interval: List[Timestamp], T_0: float):
    interval = [ts.to_secs() for ts in interval]
    return [T_0 - 1 / OFF_DISTR[0] * (t - interval[0]) for t in interval]


def on_model_2(interval: List[Timestamp], T_0: float):
    interval = [ts.to_secs() for ts in interval]
    return [T_0 + ON_DISTR[0] * (t - interval[0]) for t in interval]


on_fc = FlowCondition(0, on_model)
off_fc = FlowCondition(1, off_model)
on_fc2 = FlowCondition(2, on_model_2)
off_fc2 = FlowCondition(3, off_model_2)

models: List[FlowCondition] = [on_fc, off_fc]

if CS_VERSION in [1]:
    on_event = Event('', 'on', 'h_0')
    off_event = Event('', 'off', 'c_0')
    events = [on_event, off_event]
if CS_VERSION in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    on_event = Event('!open', 'on', 'h_0')
    off_event = Event('!open', 'off', 'c_0')
    on_event2 = Event('open', 'on', 'h_1')
    off_event2 = Event('open', 'off', 'c_1')
    events = [on_event, off_event, on_event2, off_event2]
if CS_VERSION in [3, 8, 9, 10]:
    on_event3 = Event('open2', 'on', 'h_2')
    off_event3 = Event('open2', 'off', 'c_2')
    events += [on_event3, off_event3]
if CS_VERSION in [8]:
    models += [off_fc2]
if CS_VERSION in [8, 9, 10]:
    models += [on_fc2]

model_to_distr = {}
for m in models:
    model_to_distr[m.f_id] = []

temperature = RealValuedVar(models, [], model_to_distr, label='T_r')
args = {'name': 'thermostat', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
thermostat_cs = SystemUnderLearning([temperature], events, parse_data, label_event, get_thermo_param, is_chg_pt,
                                    args=args)

test = False
if test:
    # test event configuration
    print(thermostat_cs.symbols)

    # test trace processing
    thermo_traces = [file for file in os.listdir('./resources/traces/uppaal') if file.startswith('THERMO')]
    for trace in thermo_traces:
        thermostat_cs.process_data('./resources/traces/uppaal/' + trace)

    # test visualization
    for t in thermostat_cs.traces:
        print(t)
        # thermostat_cs.plot_trace(title='test', xlabel='time [s]', ylabel='degrees C°')

    # test segment identification
    segments = thermostat_cs.get_segments(Trace([on_event]))
    print(len(segments))

    # test model identification query
    teacher = Teacher(thermostat_cs)
    print(teacher.mi_query(Trace([])))

    # test hypothesis testing query
    metrics = [get_thermo_param(s, on_fc) for s in segments]
    print(metrics)
    print(thermostat_cs.vars[0].model2distr[0])
    print(teacher.ht_query(Trace([on_event]), on_fc, save=True))
    print(thermostat_cs.vars[0].model2distr[0])
    print(thermostat_cs.vars[0].model2distr[1])
    print(teacher.ht_query(Trace([on_event, off_event]), off_fc, save=True))
    print(thermostat_cs.vars[0].model2distr[1])
    # thermostat_cs.plot_distributions()
