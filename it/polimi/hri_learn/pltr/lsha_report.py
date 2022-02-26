import configparser
import sys

from it.polimi.hri_learn.lstar_sha.learner import ObsTable

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

SAVE_PATH = config['SUL CONFIGURATION']['REPORT_SAVE_PATH']


def save_data(symbols, distr, obstable: ObsTable, traces, time, sha_name):
    f = open(SAVE_PATH + sha_name + '.txt', 'w')
    content = ''
    # Report file structure:
    # [MONITORED EVENTS AND SYMBOLS]
    content += '--OBSERVABLE EVENTS--\n\n'
    for sym in symbols.keys():
        content += '{}: {}'.format(sym, symbols[sym]) + '\n'

    # [LEARNED DISTRIBUTIONS]
    content += '\n\n--LEARNED DISTRIBUTIONS--\n\n'
    for (i, d) in enumerate(distr):
        mu: float = d[0]
        sigma: float = d[1]
        content += 'N_{}({:.6f}, {:.6f})\n'.format(i, mu, sigma)

    # [FINAL OBS TABLE]
    content += '\n\n--FINAL OBSERVATION TABLE--\n\n'
    content += obstable.to_str(filter_empty=True) + '\n'

    content += '\n\n--PERFORMANCE DATA--\n\n'
    # [SAMPLED TRACES]
    content += 'SAMPLED TRACES:\t{}\n'.format(traces)
    # [RUNNING TIME]
    content += 'RUNNING TIME:\t{}\n'.format(time)

    f.write(content)
    f.close()
