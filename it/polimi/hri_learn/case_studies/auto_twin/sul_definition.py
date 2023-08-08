import configparser
from typing import List

import src.ekg_extractor.mgrs.db_connector as conn
from it.polimi.hri_learn.case_studies.auto_twin.sul_functions import label_event, parse_data, get_rand_param, \
    is_chg_pt
from it.polimi.hri_learn.domain.lshafeatures import Event, ProbDistribution
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
foo_distr = ProbDistribution(0, {'avg': 0.0})

model2distr = {0: []}
s_id = RealValuedVar([foo_fc], [], model2distr, label='s_id')

# define events
driver = conn.get_driver()
querier: Ekg_Querier = Ekg_Querier(driver)
unique_events = querier.get_unique_events()
events: List[Event] = [Event('', e, e.lower()) for e in unique_events]

DRIVER_SIG = ['s_id']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'auto_twin', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
auto_twin_cs = SystemUnderLearning([s_id], events, parse_data, label_event, get_rand_param, is_chg_pt, args=args)

conn.close_connection(driver)
test = False
if test:
    driver = conn.get_driver()
    querier: Ekg_Querier = Ekg_Querier(driver)

    test_entities = [1, 2]
    evt_seqs = []
    for entity in test_entities:
        evt_seqs.append(querier.get_events_by_entity(str(entity)))

    for seq in evt_seqs:
        auto_twin_cs.process_data(seq)
        print(auto_twin_cs.traces[-1])

    conn.close_connection(driver)
