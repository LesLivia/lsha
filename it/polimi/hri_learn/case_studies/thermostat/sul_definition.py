import math
from typing import List

from it.polimi.hri_learn.case_studies.thermostat.sul_functions import label_event, parse_data, get_thermo_param
from it.polimi.hri_learn.domain.lshafeatures import RealValuedVar, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import Event, Timestamp
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning
from it.polimi.hri_learn.lstar_sha.teacher import Teacher

CLOSED_R = 100.0
OFF_DISTR = (100.0, 1.0, 200)
ON_DISTR = (0.7, 0.01, 200)
DRIVER_SIG = 't.ON'
DEFAULT_M = 1
DEFAULT_DISTR = 1


def off_model(interval: List[Timestamp], T_0: float):
    interval = [ts.to_secs() for ts in interval]
    return [T_0 * math.exp(-1 / OFF_DISTR[0] * (t - interval[0])) for t in interval]


def on_model(interval: List[Timestamp], T_0: float):
    interval = [ts.to_secs() for ts in interval]
    coeff = CLOSED_R * ON_DISTR[0]
    return [coeff - (coeff - T_0) * math.exp(-(1 / CLOSED_R) * (t - interval[0])) for t in interval]


on_fc = FlowCondition(0, on_model)
off_fc = FlowCondition(1, off_model)

model_to_distr = {on_fc.f_id: [], off_fc.f_id: []}
temperature = RealValuedVar([on_fc, off_fc], [], model_to_distr, label='T_r')

on_event = Event('', 'on', 'h_0')
off_event = Event('', 'off', 'c_0')

args = {'name': 'thermostat', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
thermostat_cs = SystemUnderLearning([temperature], [on_event, off_event],
                                    parse_data, label_event, get_thermo_param, args=args)

test = False
if test:
    # test event configuration
    print(thermostat_cs.symbols)

    # test trace processing
    thermostat_cs.process_data('./resources/traces/uppaal/THERMO_1.txt')

    # test visualization
    for t in thermostat_cs.traces:
        print(t)
        thermostat_cs.plot_trace(title='test', xlabel='time [s]', ylabel='degrees C°')

    # test segment identification
    segments = thermostat_cs.get_segments('h_0')
    print(len(segments))

    # test model identification query
    teacher = Teacher(thermostat_cs)
    print(teacher.mi_query(''))

    # test hypothesis testing query
    metrics = [get_thermo_param(s, on_fc) for s in segments]
    print(metrics)
    print(thermostat_cs.vars[0].model2distr[0])
    print(teacher.ht_query('h_0', on_fc, save=True))
    print(thermostat_cs.vars[0].model2distr[0])
    print(thermostat_cs.vars[0].model2distr[1])
    print(teacher.ht_query('h_0c_0', off_fc, save=True))
    print(thermostat_cs.vars[0].model2distr[1])
    thermostat_cs.plot_distributions()
