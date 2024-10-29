import configparser
import os
from typing import List

import skg_main.skg_mgrs.connector_mgr as conn
from skg_main.skg_mgrs.skg_reader import Skg_Reader
from skg_main.skg_model.schema import Timestamp as skg_Timestamp
from skg_main.skg_model.semantics import EntityForest

from sha_learning.case_studies.auto_twin.sul_functions import label_event, parse_data, get_rand_param, \
    is_chg_pt
from sha_learning.domain.lshafeatures import Event, ProbDistribution
from sha_learning.domain.sigfeatures import Timestamp as lsha_Timestamp
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from sha_learning.learning_setup.teacher import Teacher

config = configparser.ConfigParser()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']

DRIVER_SIG = ['s_id']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'auto_twin', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}

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

act_to_sensors = dict()
auto_twin_cs = SystemUnderLearning([], [], parse_data, label_event, get_rand_param, is_chg_pt, args=args)


def getSUL():
    # define events
    driver = conn.get_driver()
    reader: Skg_Reader = Skg_Reader(driver)
    unique_events = reader.get_activities()

    # FIXME should be generic
    if 'Dirty' in [e.act.split(' ')[0] for e in unique_events]:
        act_to_sensors = {"Dirty Material Input": "S1",
                          "Assembled production": "S2",
                          "Composition of charges": "S3",
                          "Sterilizer Load Released": "S4",
                          "Commissioner": "S5",
                          "Load L+D released": "S6",
                          "Assembly": "S7",
                          "Loaded on L+D trolley": "S8",
                          "L+D Load Started": "S9",
                          "Enter Cleaning": "S10",
                          "Enter Storage": "S11",
                          "Exit Storage": "S12"}
    elif 'Read Lock Status' in [e.act for e in unique_events]:
        act_to_sensors = {'Pass Sensor S1': 'S1', 'Pass Sensor S2': 'S2', 'Pass Sensor S3': 'S3',
                          'Pass Sensor S4': 'S4',
                          'Pass Sensor S5': 'S5', 'Pass Sensor S6': 'S6', 'Pass Sensor S101': 'S101',
                          'Pass Sensor S105': 'S105', 'Pass Sensor S100': 'S100', 'Pass Sensor S7': 'S7',
                          'Pass Sensor S8': 'S8', 'Pass Sensor S102': 'S102', 'Pass Sensor S104': 'S104',
                          'Pass Sensor S9': 'S9', 'Pass Sensor S10': 'S10', 'Pass Sensor S103': 'S103',
                          'Pass Sensor S11': 'S11', 'Pass Sensor S12': 'S12', 'Pass Sensor S13': 'S13',
                          'Pass Sensor S14': 'S14', 'Pass Sensor S15': 'S15', 'Pass Sensor S16': 'S16',
                          'Pass Sensor S17': 'S17', 'Start Break': 'S200', 'Stop Break': 'S201',
                          'Read Lock Status': 'S202', 'Read WIP amount': 'S203', 'Pass Sensor S106': 'S106',
                          'Pass Sensor CS001': 'S1', 'Pass Sensor CS002': 'S2', 'Pass Sensor CS003': 'S3',
                          'Pass Sensor CS004': 'S4', 'Pass Sensor CS005': 'S5', 'Pass Sensor CS006': 'S6',
                          'Pass Sensor CS101': 'S101', 'Pass Sensor CS105': 'S105', 'Pass Sensor CS100': 'S100',
                          'Pass Sensor CS007': 'S7', 'Pass Sensor CS008': 'S8', 'Pass Sensor CS102': 'S102',
                          'Pass Sensor CS104': 'S104', 'Pass Sensor CS009': 'S9', 'Pass Sensor CS010': 'S10',
                          'Pass Sensor CS4102': 'S4102', 'Pass Sensor CS4103': 'S4103',
                          'Pass Sensor CS4104': 'S4104', 'Pass Sensor CS4201': 'S4201',
                          'Pass Sensor CS4202': 'S4202', 'Pass Sensor CS4301': 'S4301',
                          'Pass Sensor CS4401': 'S4401', 'Pass Sensor CS106': 'S106', 'Pass Sensor CS011': 'S11',
                          'Pass Sensor CS012': 'S12', 'Pass Sensor CS013': 'S13', 'Pass Sensor CS014': 'S14',
                          'Pass Sensor CS015': 'S15', 'Pass Sensor CS016': 'S16', 'Pass Sensor CS4101': 'S4101',
                          'Pass Sensor CS4203': 'S4203', 'Pass Sensor CS4204': 'S4204', 'Pass Sensor VS001': 'S501',
                          'Pass Sensor VS002': 'S502', 'Pass Sensor VS003': 'S503',
                          'Read power': 'S601', 'Read temperature': 'S602'}
    else:
        act_to_sensors = {'Pass Sensor LOAD_1': 'S11', 'Pass Sensor PROCESS_1': 'S12', 'Pass Sensor UNLOAD_1': 'S13',
                          'Pass Sensor LOAD_2': 'S21', 'Pass Sensor PROCESS_2': 'S22', 'Pass Sensor UNLOAD_2': 'S23',
                          'Pass Sensor FAIL_1': 'S14', 'Pass Sensor BLOCK_2': 'S24', 'Pass Sensor LOAD_3': 'S31',
                          'Pass Sensor PROCESS_3': 'S32', 'Pass Sensor LOAD_4': 'S41', 'Pass Sensor UNLOAD_3': 'S33',
                          'Pass Sensor PROCESS_4': 'S42', 'Pass Sensor LOAD_5': 'S51', 'Pass Sensor PROCESS_5': 'S52',
                          'Pass Sensor UNLOAD_5': 'S53', 'Pass Sensor UNLOAD_4': 'S43', 'Pass Sensor FAIL_5': 'S54',
                          'Pass Sensor BLOCK_1': 'S15', 'Pass Sensor BLOCK_5': 'S55', 'Pass Sensor BLOCK_3': 'S34',
                          'Pass Sensor BLOCK_4': 'S44'}

    for e in unique_events:
        if e.act in act_to_sensors:
            e.act = act_to_sensors[e.act]

    events: List[Event] = [Event('', e.act, e.act.lower()) for e in
                           unique_events]

    if pov == 'plant':
        labels_hierarchy = reader.get_entity_labels_hierarchy()
        resources = reader.get_resources(labels_hierarchy=labels_hierarchy)
    else:
        resources = []

    vars = [s_id] if pov != 'plant' else [state_sig, s_id]
    auto_twin_cs = SystemUnderLearning(vars, events, parse_data, label_event, get_rand_param, is_chg_pt, args=args)

    conn.close_connection(driver)
    return auto_twin_cs, act_to_sensors


test = False
if test:
    driver = conn.get_driver()
    reader: Skg_Reader = Skg_Reader(driver)

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
        TEST_N = 17

        if config['AUTO-TWIN CONFIGURATION']['POV'].lower() == 'item':
            entities = reader.get_items(labels_hierarchy=reader.get_entity_labels_hierarchy(), limit=TEST_N,
                                        random=True)
        else:
            entities = reader.get_resources(labels_hierarchy=reader.get_resource_labels_hierarchy(), limit=TEST_N,
                                            random=True)

        for entity in entities[:TEST_N]:
            if pov == 'item':
                entity_tree = reader.get_entity_tree(entity.entity_id, EntityForest([]), reverse=True)
                events = reader.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
            elif pov == 'resource':
                entity_tree = reader.get_entity_tree(entity.entity_id, EntityForest([]))
                events = reader.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
            if len(events) > 0:
                evt_seqs.append(events)
    else:
        events = reader.get_events_by_timestamp(START_T, END_T)
        entity_tree = reader.get_entity_tree("Oven", EntityForest([]))
        events = reader.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
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
