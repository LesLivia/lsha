import configparser
import math
import sys
import warnings
from datetime import datetime
from typing import List

import it.polimi.hri_learn.pltr.ha_pltr as ha_pltr
from it.polimi.hri_learn.lstar_sha.learner import Learner
from it.polimi.hri_learn.lstar_sha.logger import Logger
from it.polimi.hri_learn.lstar_sha.teacher import Teacher
import it.polimi.hri_learn.pltr.lsha_report as report

# LEARNING PROCEDURE SETUP
warnings.filterwarnings('ignore')
startTime = datetime.now()

config = configparser.ConfigParser()
config.sections()
config.read(sys.argv[1])
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']
CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'][0])
RESAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY']

N_0 = (0.003, 0.0001, 100)
N_1 = (0.004, 0.0004, 100)

LOGGER = Logger()
PROB_DISTR = [N_0, N_1]

if CS_VERSION in [1]:
    UNCONTR_EVTS = {}
elif CS_VERSION in [2]:
    UNCONTR_EVTS = {'w': 'in_waiting_room'}  # , 'o': 'in_office'}
elif CS_VERSION in [3, 4, 5]:
    UNCONTR_EVTS = {'w': 'in_waiting_room', 'o': 'in_office'}
elif CS_VERSION == 'x':
    UNCONTR_EVTS = {'s': 'sat', 'r': 'ran', 'h': 'harsh_env', 'l': 'load', 'a': 'assisted_walk'}

CONTR_EVTS = {'u': 'start_moving', 'd': 'stop_moving'}


def idle_model(interval: List[float], F_0: float):
    return [F_0 * math.exp(-N_0[0] * (t - interval[0])) for t in interval]


def busy_model(interval: List[float], F_0: float):
    return [1 - (1 - F_0) * math.exp(-N_1[0] * (t - interval[0])) for t in interval]


MODELS = [idle_model, busy_model]

TEACHER = Teacher(MODELS, PROB_DISTR)
TEACHER.compute_symbols(list(UNCONTR_EVTS.keys()), list(CONTR_EVTS.keys()))
for sym in TEACHER.get_symbols().keys():
    print('{}: {}'.format(sym, TEACHER.get_symbols()[sym]))

LEARNER = Learner(TEACHER)

# RUN LEARNING ALGORITHM:
LEARNED_HA = LEARNER.run_lsha(filter_empty=True)

# PLOT (AND SAVE) RESULT
HA_SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH']

SHA_NAME = '{}_{}_{}'.format(CS, RESAMPLE_STRATEGY, CS_VERSION)
graphviz_sha = ha_pltr.to_graphviz(LEARNED_HA, SHA_NAME, HA_SAVE_PATH, view=True)

if config['DEFAULT']['PLOT_DISTR'] == 'True':
    TEACHER.plot_distributions()

report.save_data(TEACHER.get_symbols(), TEACHER.get_distributions(), LEARNER.get_table(),
                 len(TEACHER.get_signals()), datetime.now() - startTime, SHA_NAME)
print('----> EXPERIMENTAL RESULTS SAVED IN: {}{}.txt'.format(config['SUL CONFIGURATION']['REPORT_SAVE_PATH'], SHA_NAME))
