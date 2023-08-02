import configparser
import os
from typing import List

import src.ekg_extractor.mgrs.db_connector as conn
from it.polimi.hri_learn.case_studies.auto_twin.sul_functions import label_event, parse_data, get_power_param, \
    is_chg_pt
from it.polimi.hri_learn.domain.lshafeatures import Event, NormalDistribution
from it.polimi.hri_learn.domain.sigfeatures import Timestamp
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from src.ekg_extractor.mgrs.ekg_queries import Ekg_Querier

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()


def foo_model(interval: List[Timestamp]):
    return interval


# define flow conditions
foo_fc: FlowCondition = FlowCondition(0, foo_model)

# define distributions
foo_distr = NormalDistribution(0, 0.0, 0.0)

model2distr = {0: []}
power = RealValuedVar([foo_fc], [], model2distr, label='P')

# define events
driver = conn.get_driver()
querier: Ekg_Querier = Ekg_Querier(driver)
unique_events = querier.get_unique_events()
events: List[Event] = [Event('', e[0], e[0].lower()) for e in unique_events]

DRIVER_SIG = ['w', 'pr']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'energy', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
auto_twin_cs = SystemUnderLearning([power], events, parse_data, label_event, get_power_param, is_chg_pt, args=args)

conn.close_connection(driver)
test = False
if test:
    TEST_PATH = '/Users/lestingi/PycharmProjects/lsha/resources/traces/simulations/ENERGY/'
    traces_files = os.listdir(TEST_PATH)
    traces_files = [file for file in traces_files if file.startswith('_')]
