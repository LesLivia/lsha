import configparser
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from sha_learning.domain.lshafeatures import TimedTrace, Event
from sha_learning.domain.sigfeatures import SampledSignal

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', '')[0])
SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH']


def double_plot(signal1: SampledSignal, signal2: SampledSignal, signal3: SampledSignal, t: TimedTrace,
                title: str, events_cs: List[Event]):
    fig, axs = plt.subplots(3, figsize=(40, 30))

    timestamps1 = [pt.timestamp for pt in signal1.points]
    timestamps2 = [pt.timestamp for pt in signal2.points]
    timestamps3 = [pt.timestamp for pt in signal3.points]
    v1 = [pt.value for pt in signal1.points]
    v2 = [pt.value for pt in signal2.points]
    v3 = [pt.value for pt in signal3.points]

    t1 = [x.to_secs() for x in timestamps1]
    axs[0].plot(t1, v1, label='fatigue')
    axs[0].plot(t1, [0] * len(v1), '--', color='grey', linewidth=.5)

    t2 = [x.to_secs() for x in timestamps2]
    axs[1].plot(t2, v2, label='busy')
    axs[1].plot(t2, [0] * len(v2), '--', color='grey', linewidth=.5)

    t3 = [x.to_secs() for x in timestamps3]
    axs[2].plot(t3, v3, label='busy')
    axs[2].plot(t3, [0] * len(v3), '--', color='grey', linewidth=.5)

    HEIGHT1 = max(v1)
    HEIGHT2 = max(v2)
    HEIGHT3 = max(v3)

    colors = ['g', 'r', 'orange', 'purple']

    marker = 'x'
    height1 = HEIGHT1
    height2 = HEIGHT2
    height3 = HEIGHT3

    i = 0
    labels = [e.symbol for e in t.e]
    events = [ts.to_secs() for ts in t.t]
    axs[0].plot(events, [height1] * len(events), marker, color=colors[i], label=labels[i])
    axs[0].vlines(events, [0] * len(events), [height1] * len(events), color=colors[i], linewidth=0.5)
    for i, e in enumerate(events):
        text = [x.guard + ', ' + x.chan for x in events_cs if x.symbol == labels[i]][0]
        axs[0].text(e, height1, text, fontsize=20)

    i = 0
    axs[1].plot(events, [height2] * len(events), marker, color=colors[i], label=labels[i])
    axs[1].vlines(events, [0] * len(events), [height2] * len(events), color=colors[i], linewidth=0.5)
    for i, e in enumerate(events):
        text = [x.guard + ', ' + x.chan for x in events_cs if x.symbol == labels[i]][0]
        axs[1].text(e, height2, text, fontsize=20)

    i = 0
    axs[2].plot(events, [height3] * len(events), marker, color=colors[i], label=labels[i])
    axs[2].vlines(events, [0] * len(events), [height3] * len(events), color=colors[i], linewidth=0.5)
    for i, e in enumerate(events):
        text = [x.guard + ', ' + x.chan for x in events_cs if x.symbol == labels[i]][0]
        axs[2].text(e, height3, text, fontsize=20)

    step = 30
    xticks = [str(x.to_secs()) for x in timestamps1][::step]
    axs[0].set_xticks(ticks=t1[::step])
    axs[0].set_xticklabels(labels=xticks, fontsize=24)
    yticks = np.arange(0, max(v1), 0.05)
    axs[0].set_yticks(ticks=yticks)
    axs[0].set_yticklabels(labels=['{:.1f}'.format(l) for l in yticks], fontsize=24)

    step = 30
    xticks = [str(x.to_secs()) for x in timestamps2][::step]
    axs[1].set_xticks(ticks=t2[::step])
    axs[1].set_xticklabels(labels=xticks, fontsize=24)
    yticks = np.arange(min(v2), max(v2), 2)
    axs[1].set_yticks(ticks=yticks)
    axs[1].set_yticklabels(labels=['{:.1f}'.format(l) for l in yticks], fontsize=24)

    step = 30
    xticks = [str(x.to_secs()) for x in timestamps3][::step]
    axs[2].set_xticks(ticks=t3[::step])
    axs[2].set_xticklabels(labels=xticks, fontsize=24)
    yticks = np.arange(min(v3), max(v3), 2)
    axs[2].set_yticks(ticks=yticks)
    axs[2].set_yticklabels(labels=['{:.1f}'.format(l) for l in yticks], fontsize=24)

    axs[0].set_xlim(t1[0], t1[-1])
    axs[0].set_xlabel('t [hh:mm]', fontsize=24)
    axs[0].set_ylabel('F [%]', fontsize=24)

    axs[1].set_xlim(t2[0], t2[-1])
    axs[1].set_xlabel('t [hh:mm]', fontsize=24)
    axs[1].set_ylabel('Human Position X [m]', fontsize=24)

    axs[2].set_xlim(t3[0], t3[-1])
    axs[2].set_xlabel('t [hh:mm]', fontsize=24)
    axs[2].set_ylabel('Human Position Y [m]', fontsize=24)

    # axs[0].legend(fontsize=20)
    # axs[1].legend(fontsize=20)

    fig.savefig((SAVE_PATH + '{}.pdf').format(os.environ['LSHA_RES_PATH'], title))

    del fig, axs
