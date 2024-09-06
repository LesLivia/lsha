import configparser
import os
import random
import subprocess
from typing import List, Set, Dict

import skg_main.skg_mgrs.connector_mgr as conn
from sha_learning.domain.lshafeatures import Trace, Event
from sha_learning.learning_setup.logger import Logger
from skg_main.skg_mgrs.skg_reader import Skg_Reader
from skg_main.skg_model.schema import Entity
from skg_main.skg_model.schema import Timestamp as skg_Timestamp
from skg_main.skg_model.semantics import EntityForest, EntityTree

config = configparser.ConfigParser()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']
CS_VERSION = config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', '')

UPP_EXE_PATH = config['TRACE GENERATION']['UPPAAL_PATH']
UPP_OUT_PATH = config['TRACE GENERATION']['UPPAAL_OUT_PATH']
SCRIPT_PATH = config['TRACE GENERATION']['UPPAAL_SCRIPT_PATH']

SIM_LOGS_PATH = config['TRACE GENERATION']['SIM_LOGS_PATH']
LOG_FILES = ['humanFatigue.log', 'humanPosition.log']

if CS == 'HRI':
    LINE_1 = ['bool force_exe = true;\n', 'bool force_exe']
    LINE_2 = ['int force_act[MAX_E] = ', 'int force_act']
    LINE_3 = ['const int TAU = {};\n', 'const int TAU']
    LINE_4 = ['amy = HFoll_{}(1, 48, 2, 3, -1);\n', 'amy = HFoll_']
    LINE_5 = ['const int VERSION = {};\n', 'const int VERSION']
    LINES_TO_CHANGE = [LINE_1, LINE_2, LINE_3, LINE_4, LINE_5]
else:
    LINE_1 = ['bool force_exe = true;\n', 'bool force_exe']
    LINE_2 = ['int force_open[MAX_E] = ', 'int force_open']
    LINE_3 = ['const int TAU = {};\n', 'const int TAU']
    LINE_4 = ['r = Room_{}(15.2);\n', 'r = Room']
    LINES_TO_CHANGE = [LINE_1, LINE_2, LINE_3, LINE_4]

UPP_MODEL_PATH = config['TRACE GENERATION']['UPPAAL_MODEL_PATH']
UPP_QUERY_PATH = config['TRACE GENERATION']['UPPAAL_QUERY_PATH'].format(CS_VERSION)

RESAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY']

MAX_E = 15
LOGGER = Logger('TRACE GENERATOR')


class TraceGenerator:
    def __init__(self, word: Trace = Trace([]), pov: str = None,
                 start_dt: str = None, end_dt: str = None, start_ts: str = None, end_ts: str = None):
        self.word = word
        self.events: List[Event] = word.events
        self.evt_int: List[int] = []

        self.ONCE = False
        self.processed_traces: Set[str] = set()

        if RESAMPLE_STRATEGY == 'SKG':
            self.labels_hierarchy: List[List[str]] = []
            self.processed_entities: Dict[Entity, EntityTree] = {}
            self.pov = pov
            self.start_dt = start_dt
            self.end_dt = end_dt
            self.start_ts = start_ts
            self.end_ts = end_ts

    def set_word(self, w: Trace):
        self.events = w.events
        self.evt_int = []
        self.word = w

    def evts_to_ints(self):
        for e in self.events:
            if CS == 'HRI':
                if e.symbol in ['u_2', 'u_4']:
                    self.evt_int.append(1)
                elif e.symbol in ['u_3']:
                    self.evt_int.append(3)
                elif e.symbol in ['d_3', 'd_4']:
                    self.evt_int.append(0)
                elif e.symbol in ['d_2']:
                    self.evt_int.append(2)
                else:
                    self.evt_int.append(-1)
            else:
                # for thermo example: associates a specific value
                # to variable open for each event in the requested trace
                if int(CS_VERSION) < 8:
                    if e.symbol in ['h_0', 'c_0']:
                        self.evt_int.append(0)
                    elif e.symbol in ['h_1', 'c_1']:
                        self.evt_int.append(1)
                    elif e.symbol in ['h_2', 'c_2']:
                        self.evt_int.append(2)
                else:
                    if e.symbol in ['h_0', 'c_0']:
                        self.evt_int.append(0)
                    elif e.symbol in ['h_1', 'c_1']:
                        self.evt_int.append(1)
                    elif e.symbol in ['h_2', 'c_2']:
                        self.evt_int.append(2)
                    elif e.symbol in ['h_3', 'c_3']:
                        self.evt_int.append(0)

    def get_evt_str(self):
        self.evts_to_ints()

        res = '{'
        i = 0
        for evt in self.evt_int:
            res += str(evt) + ', '
            i += 1
        while i < MAX_E - 1:
            res += '-1, '
            i += 1
        res += '-1};\n'
        return res

    def fix_model(self):
        # customized uppaal model based on requested trace
        m_r = open(UPP_MODEL_PATH, 'r')

        new_line_1 = LINE_1[0]
        values = self.get_evt_str()
        new_line_2 = LINE_2[0] + values
        tau = max(len(self.evt_int) * 50, 200)
        new_line_3 = LINE_3[0].format(tau)
        new_line_4 = LINE_4[0].format(CS_VERSION)
        new_line_5 = LINE_5[0].format(int(CS_VERSION) - 1) if CS == 'HRI' else None
        new_lines = [new_line_1, new_line_2, new_line_3, new_line_4, new_line_5]

        lines = m_r.readlines()
        found = [False] * len(new_lines)
        for line in lines:
            for (i, l) in enumerate(LINES_TO_CHANGE):
                if line.startswith(LINES_TO_CHANGE[i][1]) and not found[i]:
                    lines[lines.index(line)] = new_lines[i]
                    found[i] = True
                    break

        m_r.close()
        m_w = open(UPP_MODEL_PATH, 'w')
        m_w.writelines(lines)
        m_w.close()

    def get_traces(self, n: int = 1):
        if RESAMPLE_STRATEGY == 'UPPAAL':
            return self.get_traces_uppaal(n)
        elif RESAMPLE_STRATEGY == 'SKG':
            return self.get_traces_skg(n)
        else:
            return self.get_traces_sim(n)

    def get_traces_skg(self, n: int = 1):
        driver = conn.get_driver()
        querier: Skg_Reader = Skg_Reader(driver)

        if len(self.labels_hierarchy) == 0:
            self.labels_hierarchy = querier.get_entity_labels_hierarchy()

        if self.start_ts is not None and self.end_ts is not None:
            START_T = int(self.start_ts)
            END_T = int(self.end_ts)
        else:
            def parse_date(s: str):
                fields = s.split('-')
                return skg_Timestamp(int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]),
                                     int(fields[5]))

            START_T = parse_date(self.start_dt)
            END_T = parse_date(self.end_dt)

        evt_seqs = []
        if self.pov.lower() == 'plant':
            pov = self.pov.lower()
            entity_tree = querier.get_entity_tree("Oven", EntityForest([]))
            events = querier.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
            if len(events) > 0:
                evt_seqs.append(events)
        else:
            if self.pov.lower() == 'item':
                entities = querier.get_items(labels_hierarchy=self.labels_hierarchy, limit=n, random=True,
                                             start_t=START_T, end_t=END_T)
            else:
                entities = querier.get_resources(labels_hierarchy=querier.get_resource_labels_hierarchy(), limit=n, random=True)

            for entity in entities[:n]:
                if entity not in self.processed_entities:
                    pov = self.pov.lower()
                    if pov == 'item':
                        entity_tree = querier.get_entity_tree(entity.entity_id, EntityForest([]), reverse=True)
                        events = querier.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T,
                                                                                 pov)
                    else:
                        entity_tree = querier.get_entity_tree(entity.entity_id, EntityForest([]))
                        events = querier.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T,
                                                                                 pov)
                    if len(events) > 0:
                        evt_seqs.append(events)
                    self.processed_entities[entity] = entity_tree[0]

        conn.close_connection(driver)
        return evt_seqs

    def get_traces_sim(self, n: int = 1):
        # if self.ONCE:
        #    return []

        if CS.lower() == 'energy':
            sims = os.listdir(SIM_LOGS_PATH.format(CS))
            sims = list(filter(lambda s: s.startswith('_') and s not in self.processed_traces, sims))
            sims.sort()
        else:
            sims = os.listdir(SIM_LOGS_PATH.format(os.environ['RES_PATH'],
                                                   config['SUL CONFIGURATION']['CS_VERSION']))
            sims = list(filter(lambda s: s.startswith('SIM'), sims))
        paths = []
        for i in range(n + 1):
            if len(sims) == 0:
                break
            rand_sel = random.randint(0, 100)
            rand_sel = rand_sel % len(sims)
            if CS.lower() == 'energy':
                self.processed_traces.add(sims[rand_sel])
                paths.append(SIM_LOGS_PATH.format(CS) + '/' + sims[rand_sel])
            else:
                paths.append(SIM_LOGS_PATH.format(os.environ['RES_PATH'],
                                                  config['SUL CONFIGURATION']['CS_VERSION']) + '/' + sims[i] + '/')
        # self.ONCE = True
        return paths

    def get_traces_uppaal(self, n: int):
        # sample new traces through uppaal command line tool
        self.fix_model()
        LOGGER.debug('!! GENERATING NEW TRACES FOR: {} !!'.format(self.word))
        new_traces: List[str] = []

        for i in range(n):
            random.seed()
            n = random.randint(0, 2 ** 32)
            s = '{}_{}_{}'.format(CS, CS_VERSION, n)
            FNULL = open(os.devnull, 'w')
            p = subprocess.Popen([SCRIPT_PATH, UPP_EXE_PATH, UPP_MODEL_PATH,
                                  UPP_QUERY_PATH, str(n), UPP_OUT_PATH.format(s)], stdout=FNULL)
            p.wait()
            if p.returncode == 0:
                LOGGER.info('TRACES SAVED TO ' + s)
                # returns out file where new traces are stored
                new_traces.append(UPP_OUT_PATH.format(s))

        return new_traces
