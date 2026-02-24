import configparser
import os

os.environ['NEO4J_URI'] = 'empty'
os.environ['NEO4J_USERNAME'] = 'empty'
os.environ['NEO4J_PASSWORD'] = 'empty'
os.environ['NEO4J_SCHEMA'] = 'empty'

import warnings
from datetime import datetime

import sha_learning.pltr.lsha_report as report
import sha_learning.pltr.sha_pltr as ha_pltr
from sha_learning.case_studies.lego_factory.sul_definition import getSUL as getSUL_lego_factory
from sha_learning.case_studies.lego_factory.sul_functions import get_acquisition_bounds
from sha_learning.domain.lshafeatures import Trace
from sha_learning.domain.obstable import ObsTable
from sha_learning.domain.sulfeatures import SystemUnderLearning
from sha_learning.learning_setup.learner import Learner
from sha_learning.learning_setup.logger import Logger
from sha_learning.learning_setup.teacher import Teacher
from uppaal_generator.model_generator.sha2uppaal import generate_upp_model
from uppaal_generator.model_generator.dot2sha import parse_sha

# LEARNING PROCEDURE SETUP
warnings.filterwarnings('ignore')
startTime = datetime.now()

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + '/sha_learning/resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']
RESAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY']
LOGGER = Logger('LSHA')

SUL: SystemUnderLearning
events_labels_dict = None

SUL, events_labels_dict = getSUL_lego_factory()

TEACHER = Teacher(SUL)

long_traces = [Trace(events=[e]) for e in SUL.events]
obs_table = ObsTable([], [Trace(events=[])], long_traces)
LEARNER = Learner(TEACHER, obs_table)

# RUN LEARNING ALGORITHM:
LEARNED_HA = LEARNER.run_lsha(filter_empty=True)

# PLOT (AND SAVE) RESULT
HA_SAVE_PATH = "sha_learning/resources/learned_sha/"

SHA_NAME = '{}_{}_{}'.format(CS, RESAMPLE_STRATEGY, config['SUL CONFIGURATION']['CS_VERSION'])
graphviz_sha = ha_pltr.to_graphviz(LEARNED_HA, SHA_NAME, HA_SAVE_PATH, view=True)

# saving sha source to .txt file
sha_source = graphviz_sha.source
with open(HA_SAVE_PATH.format(os.getcwd()) + SHA_NAME + '_source.txt', 'w') as f:
    f.write(sha_source)

report.save_data(TEACHER.symbols, TEACHER.distributions, LEARNER.obs_table,
                 len(TEACHER.signals), datetime.now() - startTime, SHA_NAME, events_labels_dict,
                 os.getcwd())
LOGGER.info(
    '----> EXPERIMENTAL RESULTS SAVED IN: {}{}.txt'.format(config['SUL CONFIGURATION']['REPORT_SAVE_PATH'], SHA_NAME))

config = configparser.ConfigParser()
config.read('uppaal_generator/resources/config.ini')
config.sections()
AUTOMATON_NAME = SHA_NAME
AUTOMATON_START, AUTOMATON_END = get_acquisition_bounds()
AUTOMATON_PATH = config['AUTOMATON']['automaton.graph.path'].format(AUTOMATON_NAME)

sha = parse_sha(AUTOMATON_PATH, AUTOMATON_NAME)
model_path = generate_upp_model(sha, AUTOMATON_NAME, AUTOMATON_START, AUTOMATON_END)
