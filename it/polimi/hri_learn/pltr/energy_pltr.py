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


def double_plot(timestamps1, v1, timestamps2, v2, t: TimedTrace, title, filtered=False, timestamps3=None, v3=None):
    subplots = 2 if timestamps3 is None else 3
    fig, axs = plt.subplots(subplots, figsize=(50, 30))

    SIG_WIDTH = 2.0

    t1 = [x.to_secs() for x in timestamps1]
    axs[0].plot(t1, v1, 'k-', label='power', linewidth=SIG_WIDTH)
    axs[0].plot(t1, [0] * len(v1), 'k--', linewidth=.5)

    t2 = [x.to_secs() for x in timestamps2]
    axs[1].plot(t2, v2, 'k-', label='speed', linewidth=SIG_WIDTH)
    axs[1].plot(t2, [0] * len(v2), 'k-', linewidth=.5)

    if timestamps3 is not None:
        t3 = [x.to_secs() for x in timestamps3]
        axs[2].plot(t3, v3, 'k-', label='pressure', linewidth=SIG_WIDTH)
        axs[2].plot(t3, [0] * len(v3), 'k-', linewidth=.5)

    LABEL_FONT = 28
    TICK_FONT = 22
    EVENT_FONT = 22
    EVENT_WIDTH = 2.0
    TITLE_FONT = 38
    MARKER_SIZE = 30

    HEIGHT1 = max(v1) + 1
    HEIGHT2 = max(v2) + 100
    HEIGHT3 = max(v3) + 10

    colors = ['orange', 'b', 'green', 'red']
    labels = ['spindle start', 'spindle stop', 'pressure up', 'pressure down']

    marker = 'x'
    height1 = HEIGHT1
    height2 = HEIGHT2
    height3 = HEIGHT3

    i = 0
    labels = [e.symbol for e in t.e]
    events = [ts.to_secs() for ts in t.t]
    # axs[0].vlines(events, [0] * len(events), [height1] * len(events), color=colors[i], linewidth=EVENT_WIDTH)
    for i, e in enumerate(events):
        if labels[i] == 'l':
            color = colors[2]
            marker = '^'
        elif labels[i] == 'u':
            color = colors[3]
            marker = 'v'
        elif labels[i] == 'i_0':
            color = colors[1]
            marker = 'v'
        else:
            color = colors[0]
            marker = '^'
        axs[0].plot(e, height1, marker, color=color, label=labels[i], markersize=MARKER_SIZE)
        axs[0].vlines(e, 0, height1, color='k', linewidth=EVENT_WIDTH)
        # axs[0].text(e, height1, labels[i], fontsize=EVENT_FONT)

    i = 0
    # axs[1].plot(events, [height2] * len(events), marker, color=colors[i], label=labels[i])
    # axs[1].vlines(events, [0] * len(events), [height2] * len(events), color=colors[i], linewidth=EVENT_WIDTH)
    for i, e in enumerate(events):
        if labels[i] in ['l', 'u']:
            continue

        if labels[i] == 'i_0':
            color = colors[1]
            marker = 'v'
        else:
            color = colors[0]
            marker = '^'
        axs[1].plot(e, height2, marker, color=color, label=labels[i], markersize=MARKER_SIZE)
        axs[1].vlines(e, 0, height2, color='k', linewidth=EVENT_WIDTH)
        # axs[1].text(e, height2, labels[i], fontsize=EVENT_FONT)

    i = 0
    # axs[2].plot(events, [height3] * len(events), marker, color=colors[i], label=labels[i])
    # axs[2].vlines(events, [0] * len(events), [height3] * len(events), color=colors[i], linewidth=EVENT_WIDTH)
    for i, e in enumerate(events):
        if labels[i].startswith('i') or labels[i].startswith('m'):
            continue
        if labels[i] == 'l':
            color = colors[2]
            marker = '^'
        else:
            color = colors[3]
            marker = 'v'
        axs[2].plot(e, height3, marker, color=color, label=labels[i], markersize=MARKER_SIZE)
        axs[2].vlines(e, 0, height3, color='k', linewidth=EVENT_WIDTH)
        # axs[2].text(e, height3, labels[i], fontsize=EVENT_FONT)

    step = 300
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps1][::step]
    axs[0].set_xticks(ticks=t1[::step])
    axs[0].set_xticklabels(labels=xticks, fontsize=TICK_FONT, rotation=45)
    yticks = np.arange(0, max(v1), 2)
    axs[0].set_yticks(ticks=yticks)
    axs[0].set_yticklabels(labels=yticks, fontsize=TICK_FONT)
    ymin, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0, ymax)

    step = 300
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps2][::step]
    axs[1].set_xticks(ticks=t2[::step])
    axs[1].set_xticklabels(labels=xticks, fontsize=TICK_FONT, rotation=45)
    yticks = np.arange(0, max(v2) + 200, 200)
    axs[1].set_yticks(ticks=yticks)
    axs[1].set_yticklabels(labels=yticks, fontsize=TICK_FONT)
    ymin, ymax = axs[1].get_ylim()
    axs[1].set_ylim(0, ymax)

    step = 300
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps3][::step]
    axs[2].set_xticks(ticks=t3[::step])
    axs[2].set_xticklabels(labels=xticks, fontsize=TICK_FONT, rotation=45)
    yticks = np.arange(0, max(v3) + 200, 200)
    axs[2].set_yticks(ticks=yticks)
    axs[2].set_yticklabels(labels=yticks, fontsize=TICK_FONT)
    ymin, ymax = axs[2].get_ylim()
    axs[2].set_ylim(0, ymax)

    axs[0].set_xlim(t1[0], t1[-1])
    axs[0].set_xlabel('t [hh:mm]', fontsize=LABEL_FONT)
    axs[0].set_ylabel('[kW]', fontsize=LABEL_FONT)
    axs[0].set_title('Spindle Power', fontsize=TITLE_FONT)

    axs[1].set_xlim(t1[0], t1[-1])
    axs[1].set_xlabel('t [hh:mm]', fontsize=LABEL_FONT)
    axs[1].set_ylabel('[rpm]', fontsize=LABEL_FONT)
    axs[1].set_title('Spindle Speed', fontsize=TITLE_FONT)

    axs[2].set_xlim(t1[0], t1[-1])
    axs[2].set_xlabel('t [hh:mm]', fontsize=LABEL_FONT)
    axs[2].set_ylabel('[?]', fontsize=LABEL_FONT)
    axs[2].set_title('Pallet Pressure', fontsize=TITLE_FONT)

    # axs[0].legend(fontsize=20)
    # axs[1].legend(fontsize=20)

    plt.tight_layout(pad=10.0)
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
            ax.hist(values[i][1], bins=25, density=True, histtype='step')
            with open(SAVE_PATH + '{}.txt'.format('histogram_values'), 'a') as f:
                f.write('D_{}:\n'.format(i))
                lines = [str(x) + '\n' for x in values[i][1]]
                print(lines)
                f.writelines(lines)
    fig.savefig(SAVE_PATH + '{}_{}.pdf'.format(name, 'histograms'))
    del fig
