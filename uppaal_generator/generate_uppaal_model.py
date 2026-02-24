import configparser
import os
from datetime import datetime

from uppaal_generator.model_generator.sha2uppaal import generate_upp_model

config = configparser.ConfigParser()
config.read('resources/config/config.ini')
config.sections()

SCRIPT_PATH = config['UPPAAL SETTINGS']['UPPAAL_SCRIPT_PATH']
UPPAAL_PATH = config['UPPAAL SETTINGS']['UPPAAL_PATH']
UPPAAL_OUT_PATH = config['UPPAAL SETTINGS']['UPPAAL_OUT_PATH']
BIN_W = float(config['UPPAAL SETTINGS']['HIST_W'])


def generate_uppaal_model(sha, AUTOMATON_NAME, AUTOMATON_START, AUTOMATON_END, links):
    generate_upp_model(sha, AUTOMATON_NAME, AUTOMATON_START, AUTOMATON_END, links)


def get_ts():
    ts = datetime.now()
    ts_split = str(ts).split('.')[0]
    ts_str = ts_split.replace('-', '_')
    ts_str = ts_str.replace(' ', '_')
    return ts_str


def run_exp(name, model_path, query_path):
    res_name = name + '_' + get_ts()
    os.system('{} {} {} {} {} {}'.format(SCRIPT_PATH, UPPAAL_PATH, model_path, query_path,
                                         UPPAAL_OUT_PATH.format(res_name), BIN_W))
    return UPPAAL_OUT_PATH.format(res_name)
