import math
from typing import List

from it.polimi.hri_learn.case_studies.thermostat.label_fcn import label_event
from it.polimi.hri_learn.case_studies.thermostat.parser import parse_data
from it.polimi.hri_learn.domain.lshafeatures import RealValuedVar, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import Event
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning

CLOSED_R = 100.0
OFF_DISTR = (100.0, 1.0, 200)
ON_DISTR = (0.7, 0.01, 200)
DRIVER_SIG = 't.ON'


def off_model(interval: List[float], T_0: float):
    return [T_0 * math.exp(-1 / OFF_DISTR[0] * (t - interval[0])) for t in interval]


def on_model(interval: List[float], T_0: float):
    coeff = CLOSED_R * ON_DISTR[0]
    return [coeff - (coeff - T_0) * math.exp(-(1 / CLOSED_R) * (t - interval[0])) for t in interval]


on_fc = FlowCondition(0, on_model)
off_fc = FlowCondition(1, off_model)

model_to_distr = {on_fc.f_id: [], off_fc.f_id: []}
temperature = RealValuedVar([on_fc, off_fc], [], model_to_distr, label='T_r')

on_event = Event('', 'on')
off_event = Event('', 'off')

thermostat_cs = SystemUnderLearning([temperature], [on_event, off_event], parse_data, label_event,
                                    args={'name': 'thermostat', 'driver': DRIVER_SIG})
print(thermostat_cs.symbols)
thermostat_cs.process_data('./resources/traces/uppaal/THERMO_1.txt')
for e in thermostat_cs.traces:
    print(e)
    thermostat_cs.plot_trace(title='test', xlabel='time [s]', ylabel='degrees C°')

thermostat_cs.plot_distributions()
