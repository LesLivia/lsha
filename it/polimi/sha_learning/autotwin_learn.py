import configparser
import os
import warnings
from datetime import datetime

import it.polimi.sha_learning.pltr.lsha_report as report
import it.polimi.sha_learning.pltr.sha_pltr as ha_pltr
from it.polimi.sha_learning.case_studies.auto_twin.sul_definition import auto_twin_cs, act_to_sensors
from it.polimi.sha_learning.domain.lshafeatures import Trace
from it.polimi.sha_learning.domain.obstable import ObsTable
from it.polimi.sha_learning.domain.sulfeatures import SystemUnderLearning
from it.polimi.sha_learning.learning_setup.learner import Learner
from it.polimi.sha_learning.learning_setup.teacher import Teacher
from it.polimi.sha_learning.pltr.energy_pltr import distr_hist

warnings.filterwarnings('ignore')
startTime = datetime.now()

config = configparser.ConfigParser()
if 'submodules' in os.listdir():
    curr_path = os.getcwd() + '/submodules/lsha'
else:
    curr_path = os.getcwd().split('src/lsha')[0]
config.read('{}/resources/config/config.ini'.format(curr_path))
config.sections()

CS = 'AUTO_TWIN'
CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))
RESAMPLE_STRATEGY = 'SKG'


def learn_automaton(pov: str, start: str, end: str):
    SUL: SystemUnderLearning = auto_twin_cs
    events_labels_dict = act_to_sensors

    TEACHER = Teacher(SUL, pov, start, end)

    long_traces = [Trace(events=[e]) for e in SUL.events]
    obs_table = ObsTable([], [Trace(events=[])], long_traces)
    LEARNER = Learner(TEACHER, obs_table)

    # RUN LEARNING ALGORITHM:
    LEARNED_SHA = LEARNER.run_lsha(filter_empty=True)

    # PLOT (AND SAVE) RESULT
    SHA_SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH']

    SHA_NAME = '{}_{}_{}'.format(CS, RESAMPLE_STRATEGY, pov)
    graphviz_sha = ha_pltr.to_graphviz(LEARNED_SHA, SHA_NAME, SHA_SAVE_PATH, view=True)

    # saving sha source to .txt file
    sha_source = graphviz_sha.source
    with open(SHA_SAVE_PATH + SHA_NAME + '_source.txt', 'w') as f:
        f.write(sha_source)

    if config['DEFAULT']['PLOT_DISTR'] == 'True' and config['LSHA PARAMETERS']['HT_QUERY_TYPE'] == 'S':
        distr_hist(TEACHER.hist, SHA_NAME)

    report.save_data(TEACHER.symbols, TEACHER.distributions, LEARNER.obs_table,
                     len(TEACHER.signals), datetime.now() - startTime, SHA_NAME, events_labels_dict)
    print('----> EXPERIMENTAL RESULTS SAVED IN: {}{}.txt'.format(config['SUL CONFIGURATION']['REPORT_SAVE_PATH'],
                                                                 SHA_NAME))
