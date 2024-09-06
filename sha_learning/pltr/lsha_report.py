import configparser
import os

from sha_learning.learning_setup.learner import ObsTable

config = configparser.ConfigParser()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()


def save_data(symbols, distr, obstable: ObsTable, traces, time, sha_name, events_dict=None, path=None):
    SAVE_PATH = config['SUL CONFIGURATION']['REPORT_SAVE_PATH'].format(path)
    f = open(SAVE_PATH + sha_name + '.txt', 'w')
    content = ''
    # Report file structure:
    # [EVENT LABELS DICTIONARY]
    if events_dict is not None:
        content += '--EVENT LABELS DICT--\n\n'
        for pair in events_dict.items():
            content += '{}: {}\n'.format(pair[1], pair[0])

    # [MONITORED EVENTS AND SYMBOLS]
    content += '\n\n--OBSERVABLE EVENTS--\n\n'
    for sym in symbols.keys():
        content += '{}: {}'.format(sym, symbols[sym]) + '\n'

    # [LEARNED DISTRIBUTIONS]
    content += '\n\n--LEARNED DISTRIBUTIONS--\n\n'
    for i, d in enumerate(distr[0]):
        try:
            content += 'N_{}({:.6f}, {:.6f})\n'.format(i, d.params['avg'], d.params['var'])
        except KeyError:
            content += 'D_{}({:.6f})\n'.format(i, d.params['avg'])

    # [FINAL OBS TABLE]
    content += '\n\n--FINAL OBSERVATION TABLE--\n\n'
    content += obstable.__str__(filter_empty=True) + '\n'

    content += '\n\n--PERFORMANCE DATA--\n\n'
    # [SAMPLED TRACES]
    content += 'SAMPLED TRACES:\t{}\n'.format(traces)
    # [RUNNING TIME]
    content += 'RUNNING TIME:\t{}\n'.format(time)

    f.write(content)
    f.close()
