import configparser
import warnings
from datetime import datetime

import it.polimi.hri_learn.pltr.ha_pltr as ha_pltr
import it.polimi.hri_learn.pltr.lsha_report as report
from it.polimi.hri_learn.case_studies.thermostat.sul_definition import thermostat_cs
from it.polimi.hri_learn.case_studies.energy.sul_definition import energy_cs
from it.polimi.hri_learn.lstar_sha.learner import Learner
from it.polimi.hri_learn.lstar_sha.teacher import Teacher
from it.polimi.hri_learn.domain.obstable import ObsTable
from it.polimi.hri_learn.domain.lshafeatures import Trace
from it.polimi.hri_learn.pltr.energy_pltr import distr_hist

# LEARNING PROCEDURE SETUP
warnings.filterwarnings('ignore')
startTime = datetime.now()

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']
CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'][0])
RESAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY']

SUL = energy_cs
TEACHER = Teacher(SUL)

long_traces = [Trace(events=[e]) for e in SUL.events]
obs_table = ObsTable([], [Trace(events=[])], long_traces)
LEARNER = Learner(TEACHER, obs_table)

# RUN LEARNING ALGORITHM:
LEARNED_HA = LEARNER.run_lsha(filter_empty=True)

# PLOT (AND SAVE) RESULT
HA_SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH']

SHA_NAME = '{}_{}_{}'.format(CS, RESAMPLE_STRATEGY, config['SUL CONFIGURATION']['CS_VERSION'])
graphviz_sha = ha_pltr.to_graphviz(LEARNED_HA, SHA_NAME, HA_SAVE_PATH, view=True)

# saving sha source to .txt file
sha_source = graphviz_sha.source
with open(HA_SAVE_PATH + SHA_NAME + '_source.txt', 'w') as f:
    f.write(sha_source)

if config['DEFAULT']['PLOT_DISTR'] == 'True':
    TEACHER.sul.plot_distributions()

report.save_data(TEACHER.symbols, TEACHER.distributions, LEARNER.obs_table,
                 len(TEACHER.signals), datetime.now() - startTime, SHA_NAME)
print('----> EXPERIMENTAL RESULTS SAVED IN: {}{}.txt'.format(config['SUL CONFIGURATION']['REPORT_SAVE_PATH'], SHA_NAME))

distr_hist(TEACHER.hist)
