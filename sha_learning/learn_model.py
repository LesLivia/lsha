import configparser
import os
import sys
import warnings
from datetime import datetime

import sha_learning.pltr.lsha_report as report
import sha_learning.pltr.sha_pltr as ha_pltr
from sha_learning.case_studies.auto_twin.sul_definition import auto_twin_cs, act_to_sensors
from sha_learning.case_studies.energy.sul_definition import energy_cs
from sha_learning.case_studies.energy_made.sul_definition import energy_made_cs
from sha_learning.case_studies.energy_sim.sul_definition import energy_sim_cs
from sha_learning.case_studies.hri.sul_definition import hri_cs
from sha_learning.case_studies.thermostat.sul_definition import thermostat_cs
from sha_learning.domain.lshafeatures import Trace
from sha_learning.domain.obstable import ObsTable
from sha_learning.domain.sulfeatures import SystemUnderLearning
from sha_learning.learning_setup.learner import Learner
from sha_learning.learning_setup.logger import Logger
from sha_learning.learning_setup.teacher import Teacher
from sha_learning.pltr.energy_pltr import distr_hist

# LEARNING PROCEDURE SETUP
warnings.filterwarnings('ignore')
startTime = datetime.now()

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']
RESAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY']
LOGGER = Logger('LSHA')

SUL: SystemUnderLearning
events_labels_dict = None
if CS == 'THERMO':
    SUL = thermostat_cs
elif CS == 'HRI':
    SUL = hri_cs
elif CS == 'ENERGY':
    if RESAMPLE_STRATEGY == 'SIM':
        SUL = energy_sim_cs
    elif RESAMPLE_STRATEGY == 'REAL':
        SUL = energy_cs
    elif RESAMPLE_STRATEGY == 'MADE':
        SUL = energy_made_cs
    else:
        raise RuntimeError
elif CS == 'AUTO_TWIN':
    SUL = auto_twin_cs
    events_labels_dict = act_to_sensors
else:
    raise RuntimeError

TEACHER = Teacher(SUL, pov=sys.argv[1], start_dt=sys.argv[2], end_dt=sys.argv[3])

long_traces = [Trace(events=[e]) for e in SUL.events]
obs_table = ObsTable([], [Trace(events=[])], long_traces)
LEARNER = Learner(TEACHER, obs_table)

# RUN LEARNING ALGORITHM:
LEARNED_HA = LEARNER.run_lsha(filter_empty=True)

# PLOT (AND SAVE) RESULT
HA_SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH'].format(
    os.path.abspath(__file__).split('sha_learning')[0] + 'sha_learning/')

SHA_NAME = '{}_{}_{}'.format(CS, RESAMPLE_STRATEGY, config['SUL CONFIGURATION']['CS_VERSION'])
graphviz_sha = ha_pltr.to_graphviz(LEARNED_HA, SHA_NAME, HA_SAVE_PATH, view=True)

# saving sha source to .txt file
sha_source = graphviz_sha.source
with open(HA_SAVE_PATH.format(os.getcwd()) + SHA_NAME + '_source.txt', 'w') as f:
    f.write(sha_source)

if config['DEFAULT']['PLOT_DISTR'] == 'True' and config['LSHA PARAMETERS']['HT_QUERY_TYPE'] == 'S':
    distr_hist(TEACHER.hist, SHA_NAME)

report.save_data(TEACHER.symbols, TEACHER.distributions, LEARNER.obs_table,
                 len(TEACHER.signals), datetime.now() - startTime, SHA_NAME, events_labels_dict,
                 os.getcwd())
LOGGER.info('----> EXPERIMENTAL RESULTS SAVED IN: {}{}.txt'.format(config['SUL CONFIGURATION']['REPORT_SAVE_PATH'], SHA_NAME))
