import configparser
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from sha_learning.domain.lshafeatures import TimedTrace

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH']

SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])
CS = config['SUL CONFIGURATION']['CASE_STUDY']


def double_plot(timestamps1, v1, timestamps2, v2, t: TimedTrace, title, filtered=False, timestamps3=None, v3=None):
    subplots = 2 if timestamps3 is None else 3
    fig, axs = plt.subplots(subplots, figsize=(60, 30), gridspec_kw={'height_ratios': [3, 3, 3]})

    SIG_WIDTH = 2.0

    t1 = [x.to_secs() for x in timestamps1]
    # t1 = [i for i, x in enumerate(timestamps1)]
    axs[0].plot(t1, v1, 'k-', label='power', linewidth=SIG_WIDTH)
    axs[0].plot(t1, [0] * len(v1), 'k--', linewidth=.5)

    t2 = [x.to_secs() for x in timestamps2]
    # t2 = [i for i, x in enumerate(timestamps2)]
    axs[1].plot(t2, v2, 'k-', label='speed', linewidth=SIG_WIDTH)
    axs[1].plot(t2, [0] * len(v2), 'k-', linewidth=.5)

    if timestamps3 is not None:
        t3 = [x.to_secs() for x in timestamps3]
        # t3 = [i for i, x in enumerate(timestamps3)]
        axs[2].plot(t3, v3, 'k-', label='pressure', linewidth=SIG_WIDTH)
        axs[2].plot(t3, [0] * len(v3), 'k-', linewidth=.5)

    LABEL_FONT = 32
    TICK_FONT = 30
    EVENT_FONT = 22
    EVENT_WIDTH = 2.0
    TITLE_FONT = 38
    MARKER_SIZE = 30

    HEIGHT1 = max(v1) + 1
    HEIGHT2 = max(v2) + 100
    HEIGHT3 = max(v3) + 100

    colors = ['orange', 'b', 'green', 'red']
    labels = ['spindle start', 'spindle stop', 'pressure up', 'pressure down']

    marker = 'x'
    height1 = HEIGHT1
    height2 = HEIGHT2
    height3 = HEIGHT3

    i = 0
    labels = [e.symbol for e in t.e]
    events = [ts.to_secs() for ts in t.t]
    # events = [[i for i in t1 if timestamps1[i].to_secs() == e_t.to_secs()][0] for e_t in t.t]
    axs[0].vlines(events, [0] * len(events), [height1] * len(events), color=colors[i], linewidth=EVENT_WIDTH)
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

    # axs[0].plot(events[1:-1], [0] * (len(events) - 2), color='k', linewidth=50, zorder=1)
    # ops = ['26', '14', '16', '4', '26', '2']
    # i_op = 0
    i = 0
    # axs[2].plot(events, [height2] * len(events), marker, color=colors[i], label=labels[i])
    # axs[2].vlines(events, [0] * len(events), [height2] * len(events), color=colors[i], linewidth=EVENT_WIDTH)
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
        # axs[0].vlines(e, -0.1, +0.1, color='white', linewidth=8, zorder=4)
        # if labels[i] not in ['l', 'u', 'i_0']:
        # print('{}: {}'.format(ops[i_op], e + (events[i + 1] - e)))
        # axs[0].text(e + (events[i + 1] - e) / 2, -0.05, ops[i_op], color='white', zorder=4, fontsize=40)
        # i_op += 1
        # axs[2].text(e, height2, labels[i], fontsize=EVENT_FONT)

    i = 0
    # axs[3].plot(events, [height3] * len(events), marker, color=colors[i], label=labels[i])
    # axs[3].vlines(events, [0] * len(events), [height3] * len(events), color=colors[i], linewidth=EVENT_WIDTH)
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
        # axs[3].text(e, height3, labels[i], fontsize=EVENT_FONT)

    PAD = 0.1

    step = 300
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps1][::step]
    # xticks = [str(x) for x in t1][::step] + [str(t1[-1])]
    # axs[1].set_xticks(ticks=[int(o) for o in xticks])
    axs[0].set_xticks(ticks=[x for x in t1[::step]])
    axs[0].set_xticklabels(labels=xticks, fontsize=TICK_FONT)
    xmin, xmax = axs[0].get_xlim()
    axs[0].set_xlim(xmin - PAD, xmax)
    yticks = np.arange(0, max(v1) + 1, 1)
    axs[0].set_yticks(ticks=yticks)
    axs[0].set_yticklabels(labels=yticks, fontsize=TICK_FONT)
    ymin, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0, ymax)

    # step = 60
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps2][::step]
    # axs[2].set_xticks(ticks=[int(o) for o in xticks])
    axs[1].set_xticks(ticks=[x for x in t1[::step]])
    axs[1].set_xticklabels(labels=xticks, fontsize=TICK_FONT)
    xmin, xmax = axs[1].get_xlim()
    axs[1].set_xlim(xmin - PAD, xmax)
    yticks = np.arange(0, max(v2) + 400, 400)
    axs[1].set_yticks(ticks=yticks)
    axs[1].set_yticklabels(labels=yticks, fontsize=TICK_FONT)
    ymin, ymax = axs[1].get_ylim()
    axs[1].set_ylim(0, ymax)

    # step = 120
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps3][::step]
    # axs[3].set_xticks(ticks=[int(o) for o in xticks])
    axs[2].set_xticks(ticks=[x for x in t1[::step]])
    axs[2].set_xticklabels(labels=xticks, fontsize=TICK_FONT)
    xmin, xmax = axs[2].get_xlim()
    axs[2].set_xlim(xmin - PAD, xmax)
    yticks = np.arange(0, max(v3) + 1, 800)
    axs[2].set_yticks(ticks=yticks)
    # axs[2].set_yticklabels(labels=['unlocked', 'locked'], fontsize=TICK_FONT)
    ymin, ymax = axs[2].get_ylim()
    axs[2].set_ylim(0, ymax)

    # axs[0].set_xticks(ticks=[int(o) for o in xticks])
    # axs[0].set_xticklabels(labels=xticks, fontsize=TICK_FONT)
    # axs[0].set_yticks(ticks=[0])
    # axs[0].set_yticklabels(labels=['OP'], fontsize=TICK_FONT)
    # axs[0].set_xlabel('t [s]', fontsize=LABEL_FONT)
    # axs[0].set_ylabel('', fontsize=LABEL_FONT)
    # axs[0].set_title('Operation ID', fontsize=TITLE_FONT)

    # axs[1].set_xlim(t1[0], t1[-1])
    axs[0].set_xlabel('t [hh:mm]', fontsize=LABEL_FONT)
    axs[0].set_ylabel('[kW]', fontsize=LABEL_FONT)
    axs[0].set_title('Total Power (spindle power + axis power)', fontsize=TITLE_FONT)

    # axs[2].set_xlim(t1[0], t1[-1])
    axs[1].set_xlabel('t [hh:mm]', fontsize=LABEL_FONT)
    axs[1].set_ylabel('[rpm]', fontsize=LABEL_FONT)
    axs[1].set_title('Spindle Speed', fontsize=TITLE_FONT)

    # axs[3].set_xlim(t1[0], t1[-1])
    axs[2].set_xlabel('t [hh:mm]', fontsize=LABEL_FONT)
    axs[2].set_ylabel('', fontsize=LABEL_FONT)
    axs[2].set_title('Clamping Pressure', fontsize=TITLE_FONT)

    # axs[1].legend(fontsize=20)
    # axs[2].legend(fontsize=20)

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
