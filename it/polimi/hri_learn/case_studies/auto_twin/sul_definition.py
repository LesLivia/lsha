import configparser
from typing import List

import src.ekg_extractor.mgrs.db_connector as conn
from it.polimi.hri_learn.case_studies.auto_twin.sul_functions import label_event, parse_data, get_rand_param, \
    is_chg_pt
from it.polimi.hri_learn.domain.lshafeatures import Event, ProbDistribution
from it.polimi.hri_learn.domain.sigfeatures import Timestamp as lsha_Timestamp
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from it.polimi.hri_learn.lstar_sha.teacher import Teacher
from src.ekg_extractor.mgrs.ekg_queries import Ekg_Querier
from src.ekg_extractor.model.schema import Timestamp as skg_Timestamp
from src.ekg_extractor.model.semantics import EntityForest

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']

DRIVER_SIG = ['s_id']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'auto_twin', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}

if CS == 'AUTO_TWIN':

    pov = config['AUTO-TWIN CONFIGURATION']['POV'].lower()


    def foo_model(interval: List[lsha_Timestamp]):
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
    elif unique_events[0].act.startswith('Pass'):
        act_to_sensors = {'Pass Sensor S1': 'S1', 'Pass Sensor S2': 'S2', 'Pass Sensor S3': 'S3',
                          'Pass Sensor S4': 'S4',
                          'Pass Sensor S5': 'S5', 'Pass Sensor S6': 'S6', 'Pass Sensor S101': 'S101',
                          'Pass Sensor S105': 'S105', 'Pass Sensor S100': 'S100', 'Pass Sensor S7': 'S7',
                          'Pass Sensor S8': 'S8', 'Pass Sensor S102': 'S102', 'Pass Sensor S104': 'S104',
                          'Pass Sensor S9': 'S9', 'Pass Sensor S10': 'S10', 'Pass Sensor S103': 'S103',
                          'Pass Sensor S11': 'S11', 'Pass Sensor S12': 'S12', 'Pass Sensor S13': 'S13',
                          'Pass Sensor S14': 'S14', 'Pass Sensor S15': 'S15', 'Pass Sensor S16': 'S16',
                          'Pass Sensor S17': 'S17', 'Start Break': 'S200', 'Stop Break': 'S201',
                          'Read Lock Status': 'S202', 'Read WIP amount': 'S203'}
    else:
        act_to_sensors = {'_LOAD_1': 'S11', '_PROCESS_1': 'S12', '_UNLOAD_1': 'S13',
                          '_LOAD_2': 'S21', '_PROCESS_2': 'S22', '_UNLOAD_2': 'S23',
                          '_FAIL_1': 'S14', '_BLOCK_2': 'S24', '_LOAD_3': 'S31',
                          '_PROCESS_3': 'S32', '_LOAD_4': 'S41', '_UNLOAD_3': 'S33',
                          '_PROCESS_4': 'S42', '_LOAD_5': 'S51', '_PROCESS_5': 'S52',
                          '_UNLOAD_5': 'S53', '_UNLOAD_4': 'S43', '_FAIL_5': 'S54',
                          '_BLOCK_1': 'S15', '_BLOCK_5': 'S55', '_BLOCK_3': 'S34', '_BLOCK_4': 'S44'}

    for e in unique_events:
        if e.act in act_to_sensors:
            e.act = act_to_sensors[e.act]

    events: List[Event] = [Event('', e.act, e.act.lower()) for e in
                           unique_events]

    if pov == 'plant':
        labels_hierarchy = querier.get_entity_labels_hierarchy()
        resources = querier.get_resources(labels_hierarchy=labels_hierarchy)
    else:
        resources = []

    vars = [s_id] if pov != 'plant' else [state_sig, s_id]
    auto_twin_cs = SystemUnderLearning(vars, events, parse_data, label_event, get_rand_param, is_chg_pt, args=args)

    conn.close_connection(driver)
else:
    auto_twin_cs = SystemUnderLearning([], [], parse_data, label_event, get_rand_param, is_chg_pt, args=args)

test = False
if test:
    driver = conn.get_driver()
    querier: Ekg_Querier = Ekg_Querier(driver)

    evt_seqs = []

    if 'START_T' in config['AUTO-TWIN CONFIGURATION'] and 'END_T' in config['AUTO-TWIN CONFIGURATION']:
        START_T = int(config['AUTO-TWIN CONFIGURATION']['START_T'])
        END_T = int(config['AUTO-TWIN CONFIGURATION']['END_T'])
    else:
        def parse_date(s: str):
            fields = s.split('-')
            return skg_Timestamp(int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]),
                                 int(fields[5]))


        START_T = parse_date(config['AUTO-TWIN CONFIGURATION']['START_DATE'])
        END_T = parse_date(config['AUTO-TWIN CONFIGURATION']['END_DATE'])

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
                events = querier.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
            elif pov == 'resource':
                entity_tree = querier.get_entity_tree(entity.entity_id, EntityForest([]))
                events = querier.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
            if len(events) > 0:
                evt_seqs.append(events)
    else:
        events = querier.get_events_by_timestamp(START_T, END_T)
        entity_tree = querier.get_entity_tree("Oven", EntityForest([]))
        events = querier.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
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
