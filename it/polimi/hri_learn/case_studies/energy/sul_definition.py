from typing import List

from it.polimi.hri_learn.case_studies.energy.sul_functions import label_event, parse_data, get_power_param
from it.polimi.hri_learn.domain.lshafeatures import Event, NormalDistribution
from it.polimi.hri_learn.domain.sigfeatures import Timestamp, SampledSignal
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition


# FIXME: temporarily approximated to constant function
def pwr_model(interval: List[Timestamp]):
    interval = [ts.to_secs() for ts in interval]
    AVG_PW = 1.0
    return [AVG_PW] * len(interval)


on_fc: FlowCondition = FlowCondition(0, pwr_model)

power = RealValuedVar([on_fc], [], {}, label='P')

# define events
spindle_on1 = Event('100<=w<500', 'start', 'm_0')
spindle_on2 = Event('500<=w<1000', 'start', 'm_1')
spindle_on3 = Event('1000<=w<1500', 'start', 'm_2')
spindle_on4 = Event('1500<=w<2000', 'start', 'm_3')
spindle_on5 = Event('2000<=w', 'start', 'm_3')

spindle_off = Event('', 'stop', 'i_0')

events = [spindle_off, spindle_on1, spindle_on2, spindle_on3, spindle_on4, spindle_on5]

# define distributions
off_distr = NormalDistribution(0, 0.0, 10.0)

DRIVER_SIG = 'w'
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'energy', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
energy_cs = SystemUnderLearning([power], events, parse_data, label_event, get_power_param, args=args)

test = True
if test:
    # testing data to signals conversion
    new_signals: List[SampledSignal] = parse_data(
        '/Users/lestingi/PycharmProjects/lsha/resources/traces/simulations/energy/W9_2019-10-31.csv')
