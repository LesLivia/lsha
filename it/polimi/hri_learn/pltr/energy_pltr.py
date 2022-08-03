import configparser
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from it.polimi.hri_learn.domain.lshafeatures import TimedTrace

SAVE_PATH = 'resources/learned_ha/'

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])
CS = config['SUL CONFIGURATION']['CASE_STUDY']


def double_plot(timestamps1, v1, timestamps2, v2, t: TimedTrace, title, filtered=False):
    fig, axs = plt.subplots(2, figsize=(40, 20))

    t1 = [x.to_secs() for x in timestamps1]
    axs[0].plot(t1, v1, label='power')
    axs[0].plot(t1, [0] * len(v1), '--', color='grey', linewidth=.5)

    t2 = [x.to_secs() for x in timestamps2]
    axs[1].plot(t2, v2, label='speed')
    axs[1].plot(t2, [0] * len(v2), '--', color='grey', linewidth=.5)

    HEIGHT1 = 5
    HEIGHT2 = 1000

    colors = ['g', 'r', 'orange', 'purple']
    labels = ['spindle start', 'spindle stop', 'pressure up', 'pressure down']

    marker = 'x'
    height1 = HEIGHT1
    height2 = HEIGHT2

    i = 0
    labels = [e.symbol for e in t.e]
    events = [ts.to_secs() for ts in t.t]
    axs[0].plot(events, [height1] * len(events), marker, color=colors[i], label=labels[i])
    axs[0].vlines(events, [0] * len(events), [height1] * len(events), color=colors[i], linewidth=0.5)
    for i, e in enumerate(events):
        axs[0].text(e, height1, labels[i], fontsize=12)

    i = 0
    axs[1].plot(events, [height2] * len(events), marker, color=colors[i], label=labels[i])
    axs[1].vlines(events, [0] * len(events), [height2] * len(events), color=colors[i], linewidth=0.5)
    for i, e in enumerate(events):
        axs[1].text(e, height2, labels[i], fontsize=12)

    step = 300
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps1][::step]
    axs[0].set_xticks(ticks=t1[::step])
    axs[0].set_xticklabels(labels=xticks, fontsize=24)
    yticks = np.arange(0, max(v1), 1)
    axs[0].set_yticks(ticks=yticks)
    axs[0].set_yticklabels(labels=yticks, fontsize=24)

    step = 300
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps2][::step]
    axs[1].set_xticks(ticks=t2[::step])
    axs[1].set_xticklabels(labels=xticks, fontsize=24)
    yticks = np.arange(0, max(v2), SPEED_RANGE if max(v2) < 2000 else 500)
    axs[1].set_yticks(ticks=yticks)
    axs[1].set_yticklabels(labels=yticks, fontsize=24)

    axs[0].set_xlim(t1[0], t1[-1])
    axs[0].set_xlabel('t [hh:mm]', fontsize=24)
    axs[0].set_ylabel('P [kW]', fontsize=24)

    axs[1].set_xlim(t1[0], t1[-1])
    axs[1].set_xlabel('t [hh:mm]', fontsize=24)
    axs[1].set_ylabel('w [rpwn]', fontsize=24)

    # axs[0].legend(fontsize=20)
    # axs[1].legend(fontsize=20)

    fig.savefig(SAVE_PATH + '{}.pdf'.format(title))

    del fig, axs


def distr_hist(values: Dict[int, List[float]], name: str):
    values = [(v, values[v]) for v in values]
    # values = sorted(values, key=lambda tup: sum(tup[1]) / len(tup[1]))

    fig, axs = plt.subplots(1, len(values), tight_layout=True, figsize=(5 * len(values), 5))

    for i, ax in enumerate(axs):
        ax.set_title('D_{}'.format(i))
        if CS == 'THERMO':
            ax.hist(values[i][1], bins=25)
        else:
            ax.hist(values[i][1], bins=25, density=False, histtype='step')
            with open(SAVE_PATH + '{}.txt'.format('histogram_values'), 'a') as f:
                    f.write('D_{}:\n'.format(i))
                    lines = [str(x)+'\n' for x in values[i][1]]
                    print(lines)
                    f.writelines(lines)
    fig.savefig(SAVE_PATH + '{}_{}.pdf'.format(name, 'histograms'))
    del fig
