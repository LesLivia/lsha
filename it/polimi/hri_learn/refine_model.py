import math
import sys
import warnings
from datetime import datetime
from typing import List

import it.polimi.hri_learn.pltr.ha_pltr as ha_pltr
from it.polimi.hri_learn.lstar_sha.learner import Learner
from it.polimi.hri_learn.lstar_sha.logger import Logger
from it.polimi.hri_learn.lstar_sha.teacher import Teacher

# LEARNING PROCEDURE SETUP
warnings.filterwarnings('ignore')
startTime = datetime.now()

CS_VERSION = sys.argv[2]
N_0 = (0.003, 0.0001, 100)
N_1 = (0.004, 0.0004, 100)

LOGGER = Logger()
PROB_DISTR = [N_0, N_1]

UNCONTR_EVTS = {}
if CS_VERSION == 'b':
    UNCONTR_EVTS = {'w': 'in_waiting_room'}  # , 'o': 'in_office'}
elif CS_VERSION == 'c':
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
LEARNED_HA = LEARNER.run_hl_star(filter_empty=True)
ha_pltr.plot_ha(LEARNED_HA, 'H_{}_{}{}'.format(sys.argv[1], CS_VERSION, sys.argv[3]), view=True)
TEACHER.plot_distributions()
print(datetime.now() - startTime)
