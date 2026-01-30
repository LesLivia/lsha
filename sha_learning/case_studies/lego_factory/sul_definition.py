import configparser
import os
from typing import List
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer

from sha_learning.case_studies.lego_factory.sul_functions import parse_data, label_event, is_chg_pt, get_rand_param
from sha_learning.domain.lshafeatures import Event, ProbDistribution
from sha_learning.domain.sigfeatures import Timestamp as lsha_Timestamp, Timestamp
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition

config = configparser.ConfigParser()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']

DRIVER_SIG = ['s_id']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'lego_factory', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}


def foo_model(interval: List[lsha_Timestamp]):
    return interval


# define flow conditions
foo_fc: FlowCondition = FlowCondition(0, foo_model)

# define distributions
foo_distr = ProbDistribution(0, {'avg': 0.0})

model2distr = {0: [], 1: []}
s_id = RealValuedVar([foo_fc], [], model2distr, label='s_id')
state_sig = RealValuedVar([foo_fc], [], model2distr, label='state_vec')

act_to_sensors = dict()


def getSUL():
    # define events
    ACT_TO_SENSORS = ['corner1_RETURN', 'corner1_TRANSFER', 'corner2_RETURN', 'corner2_START', 'corner2_TRANSFER',
                      'splitter1_FORWARD', 'splitter1_RETURN', 'splitter1_TRANSFER', 'splitter2_FORWARD',
                      'splitter2_RETURN', 'splitter2_TRANSFER', 'splitter3_FORWARD', 'splitter3_RETURN',
                      'splitter3_TRANSFER', 'splitter4_FORWARD', 'splitter4_RETURN', 'splitter4_TRANSFER',
                      'splitter5_CHECKOUT', 'splitter5_FINISH', 'splitter5_FORWARD', 'splitter5_RETURN',
                      'splitter5_SCRAP', 'splitter5_TRANSFER', 'station11_FAIL', 'station11_LOAD',
                      'station11_PROCESS', 'station11_TRANSFER', 'station11_UNLOAD', 'station21_FAIL',
                      'station21_LOAD', 'station21_PASS', 'station21_PROCESS', 'station21_TRANSFER',
                      'station21_UNLOAD', 'station22_FAIL', 'station22_LOAD', 'station22_PASS',
                      'station22_PROCESS', 'station22_TRANSFER', 'station22_UNLOAD', 'station31_BLOCK',
                      'station31_FAIL', 'station31_LOAD', 'station31_PASS', 'station31_PROCESS', 'station31_TRANSFER',
                      'station31_UNLOAD', 'station41_FAIL', 'station41_LOAD', 'station41_PASS', 'station41_PROCESS',
                      'station41_TRANSFER', 'station41_UNLOAD', 'station51_FAIL', 'station51_LOAD', 'station51_PASS',
                      'station51_PROCESS', 'station51_TRANSFER', 'station51_UNLOAD', 'station52_FAIL', 'station52_LOAD',
                      'station52_PASS', 'station52_PROCESS', 'station52_TRANSFER', 'station52_UNLOAD', 'station61_LOAD',
                      'station61_PASS', 'station61_PROCESS', 'station61_TRANSFER', 'station61_UNLOAD', 'station71_LOAD',
                      'station71_PASS', 'station71_PROCESS', 'station71_TRANSFER', 'station71_UNLOAD']

    events: List[Event] = [Event('', e, f's{i + 1}') for i, e in enumerate(ACT_TO_SENSORS)]

    vars = [s_id]
    lego_factory_cs = SystemUnderLearning(vars, events, parse_data, label_event, get_rand_param, is_chg_pt, args=args)

    return lego_factory_cs, act_to_sensors


SIM_LOGS_PATH = config['TRACE GENERATION']['SIM_LOGS_PATH']
os.environ['RES_PATH'] = 'TRACES/'

test = False
if test:
    sul, _ = getSUL()

    log = xes_importer.apply("/Users/livialestingi/PycharmProjects/lsha/resources/traces/event_log_processed_1.xes")
    print(f"Found {len(log)} traces")
    for trace in tqdm(log[:10]):
        sul.process_data(trace)

    for trace in sul.traces:
        print(trace)
