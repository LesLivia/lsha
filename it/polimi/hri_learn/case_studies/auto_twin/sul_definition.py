import configparser
from typing import List

import src.ekg_extractor.mgrs.db_connector as conn
from it.polimi.hri_learn.case_studies.auto_twin.sul_functions import label_event, parse_data, get_rand_param, \
    is_chg_pt
from it.polimi.hri_learn.domain.lshafeatures import Event, ProbDistribution
from it.polimi.hri_learn.domain.sigfeatures import Timestamp
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from it.polimi.hri_learn.lstar_sha.teacher import Teacher
from src.ekg_extractor.mgrs.ekg_queries import Ekg_Querier
from src.ekg_extractor.model.semantics import EntityForest

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

pov = config['AUTO-TWIN CONFIGURATION']['POV'].lower()


def foo_model(interval: List[Timestamp]):
    return interval


# define flow conditions
foo_fc: FlowCondition = FlowCondition(0, foo_model)

# define distributions
foo_distr = ProbDistribution(0, {'avg': 0.0})

model2distr = {0: [], 1: []}
s_id = RealValuedVar([foo_fc], [], model2distr, label='s_id')
if pov == 'plant':
    state_sig = RealValuedVar([foo_fc], [], model2distr, label='state_vec')

# define events
driver = conn.get_driver()
querier: Ekg_Querier = Ekg_Querier(driver)
unique_events = querier.get_activities()

# FIXME should be generic
if unique_events[0].act.startswith('Entrada'):
    act_to_sensors = {"Entrada Material Sucio": 'S1', "Cargado en carro  L+D": 'S2',
                      "Carga L+D iniciada": 'S3', "Carga L+D liberada": 'S4',
                      "Montaje": 'S5', "Producción  montada": 'S6',
                      "Composición de cargas": 'S7', "Carga de esterilizador liberada": 'S8',
                      "Carga de esterilizadorliberada": 'S9'}
    for e in unique_events:
        e.act = act_to_sensors[e.act]


events: List[Event] = [Event('', e.act.replace('Pass Sensor ', ''), e.act.replace('Pass Sensor ', '').lower()) for e in
                       unique_events]

DRIVER_SIG = ['s_id']
DEFAULT_M = 0
DEFAULT_DISTR = 0

if pov == 'plant':
    labels_hierarchy = querier.get_entity_labels_hierarchy()
    resources = querier.get_resources(labels_hierarchy=labels_hierarchy)
else:
    resources = []

args = {'name': 'auto_twin', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
vars = [s_id] if pov != 'plant' else [state_sig, s_id]
auto_twin_cs = SystemUnderLearning(vars, events, parse_data, label_event, get_rand_param, is_chg_pt, args=args)

conn.close_connection(driver)
test = False
if test:
    driver = conn.get_driver()
    querier: Ekg_Querier = Ekg_Querier(driver)

    evt_seqs = []

    START_T = int(config['AUTO-TWIN CONFIGURATION']['START_T'])
    END_T = int(config['AUTO-TWIN CONFIGURATION']['END_T'])

    if pov != 'plant':
        TEST_N = 5
        labels_hierarchy = querier.get_entity_labels_hierarchy()

        if config['AUTO-TWIN CONFIGURATION']['POV'].lower() == 'item':
            entities = querier.get_items(labels_hierarchy=labels_hierarchy, limit=TEST_N, random=True)
        else:
            entities = querier.get_resources(labels_hierarchy=labels_hierarchy, limit=TEST_N, random=True)

        for entity in entities[:TEST_N]:
            if pov == 'item':
                entity_tree = querier.get_entity_tree(entity.entity_id, EntityForest([]), reverse=True)
                events = querier.get_events_by_entity_tree(entity_tree[0], pov)
            elif pov == 'resource':
                entity_tree = querier.get_entity_tree(entity.entity_id, EntityForest([]))
                events = querier.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
            if len(events) > 0:
                evt_seqs.append(events)
    else:
        events = querier.get_events_by_timestamp(START_T, END_T)
        if len(events) > 0:
            evt_seqs.append(events)

    teacher = Teacher(auto_twin_cs)

    for seq in evt_seqs:
        auto_twin_cs.process_data(seq)
        print(auto_twin_cs.traces[-1])
        # auto_twin_cs.plot_trace(-1)
        id_cluster = teacher.ht_query(auto_twin_cs.traces[-1], foo_fc)
        print(id_cluster)

    conn.close_connection(driver)
