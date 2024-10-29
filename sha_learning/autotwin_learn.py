import configparser
import os
import warnings
from datetime import datetime

import sha_learning.pltr.lsha_report as report
import sha_learning.pltr.sha_pltr as sha_pltr
from sha_learning.case_studies.auto_twin.sul_definition import getSUL
from sha_learning.domain.lshafeatures import Trace
from sha_learning.domain.obstable import ObsTable
from sha_learning.domain.sulfeatures import SystemUnderLearning
from sha_learning.learning_setup.learner import Learner
from sha_learning.learning_setup.logger import Logger
from sha_learning.learning_setup.teacher import Teacher
from sha_learning.pltr.energy_pltr import distr_hist

warnings.filterwarnings('ignore')
startTime = datetime.now()

config = configparser.ConfigParser()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS = 'AUTO_TWIN'
CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))
RESAMPLE_STRATEGY = 'SKG'
LOGGER = Logger('LSHA')


def learn_automaton(pov: str, start_dt: str = None, end_dt: str = None,
                    start_ts: int = None, end_ts: int = None, save_path=None):
    auto_twin_cs, act_to_sensors = getSUL()
    SUL: SystemUnderLearning = auto_twin_cs
    SUL.reset_distributions()
    events_labels_dict = act_to_sensors

    TEACHER = Teacher(SUL, pov, start_dt, end_dt, start_ts, end_ts)

    long_traces = [Trace(events=[e]) for e in SUL.events]
    obs_table = ObsTable([], [Trace(events=[])], long_traces)
    LEARNER = Learner(TEACHER, obs_table)

    # RUN LEARNING ALGORITHM:
    LEARNED_SHA = LEARNER.run_lsha(filter_empty=True)

    # PLOT (AND SAVE) RESULT
    if save_path is None:
        SHA_SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH'].format(
            os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning')
    else:
        SHA_SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH'].format(save_path)

    SHA_NAME = '{}_{}_{}'.format(CS, RESAMPLE_STRATEGY, pov)
    if SHA_NAME in [file.split('-')[0] for file in os.listdir(SHA_SAVE_PATH)]:
        indexes = [file.split('-')[1][0] for file in os.listdir(SHA_SAVE_PATH) if SHA_NAME == file.split('-')[0]]
        max_index = max([int(x) for x in indexes])
        SHA_NAME += '-{}'.format(max_index + 1)
    else:
        SHA_NAME += '-0'
    graphviz_sha = sha_pltr.to_graphviz(LEARNED_SHA, SHA_NAME, SHA_SAVE_PATH, view=True)

    # saving sha source to .txt file
    sha_source = graphviz_sha.source
    with open(SHA_SAVE_PATH + SHA_NAME + '_source.txt', 'w') as f:
        f.write(sha_source)

    if config['DEFAULT']['PLOT_DISTR'] == 'True' and config['LSHA PARAMETERS']['HT_QUERY_TYPE'] == 'S':
        distr_hist(TEACHER.hist, SHA_NAME)

    report.save_data(TEACHER.symbols, TEACHER.distributions, LEARNER.obs_table,
                     len(TEACHER.signals), datetime.now() - startTime, SHA_NAME, events_labels_dict, save_path)
    LOGGER.info('----> EXPERIMENTAL RESULTS SAVED IN: {}{}.txt'.format(config['SUL CONFIGURATION']['REPORT_SAVE_PATH'],
                                                                       SHA_NAME))

    return SHA_NAME
