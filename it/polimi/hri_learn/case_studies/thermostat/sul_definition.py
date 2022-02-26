import math
from typing import List

from it.polimi.hri_learn.case_studies.thermostat.label_fcn import label_event
from it.polimi.hri_learn.case_studies.thermostat.parser import parse_data
from it.polimi.hri_learn.domain.lshafeatures import RealValuedVar, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import Event, Timestamp
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning
from it.polimi.hri_learn.lstar_sha.teacher import Teacher

CLOSED_R = 100.0
OFF_DISTR = (100.0, 1.0, 200)
ON_DISTR = (0.7, 0.01, 200)
DRIVER_SIG = 't.ON'
DEFAULT_M = 1


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

thermostat_cs = SystemUnderLearning([temperature], [on_event, off_event], parse_data, label_event,
                                    args={'name': 'thermostat', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M})

# test event correct configuration
print(thermostat_cs.symbols)
# test trace processing
thermostat_cs.process_data('./resources/traces/uppaal/THERMO_1.txt')
# test visualization
for t in thermostat_cs.traces:
    print(t)
    thermostat_cs.plot_trace(title='test', xlabel='time [s]', ylabel='degrees C°')
thermostat_cs.plot_distributions()
# test segment identification
segments = thermostat_cs.get_segments('h_0')
print(segments)
# test model identification query
teacher = Teacher(thermostat_cs)
print(teacher.mi_query('h_0c_0h_0c_0'))
