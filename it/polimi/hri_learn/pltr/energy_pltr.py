import matplotlib.pyplot as plt
import numpy as np

SAVE_PATH = './resources/plots/'


def double_plot(timestamps1, v1, timestamps2, v2, events, title, filtered=False):
    fig, axs = plt.subplots(2, figsize=(40, 20))

    t1 = [x.to_secs() for x in timestamps1]
    axs[0].plot(t1, v1, label='power')

    t2 = [x.to_secs() for x in timestamps2]
    axs[1].plot(t2, v2, label='speed')

    HEIGHT1 = 5
    HEIGHT2 = 1000

    colors = ['g', 'r', 'orange', 'purple']
    labels = ['spindle start', 'spindle stop', 'pressure up', 'pressure down']

    marker = 'x'
    height1 = HEIGHT1
    height2 = HEIGHT2

    i = 0
    events = [e.to_secs() for e in events]
    axs[0].plot(events, [height1] * len(events), marker, color=colors[i], label=labels[i])
    axs[0].vlines(events, [0] * len(events), [height1] * len(events), color=colors[i], linewidth=0.5)

    axs[1].plot(events, [height2] * len(events), marker, color=colors[i], label=labels[i])
    axs[1].vlines(events, [0] * len(events), [height2] * len(events), color=colors[i], linewidth=0.5)

    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps1][::10]
    axs[0].set_xticks(ticks=t1[::10])
    axs[0].set_xticklabels(labels=xticks, fontsize=18)
    yticks = np.arange(-HEIGHT1, max(v1), 5)
    axs[0].set_yticks(ticks=yticks)
    axs[0].set_yticklabels(labels=yticks, fontsize=18)

    step = 300 if not filtered else 5
    xticks = [str(x.hour) + ':' + str(x.min).zfill(2) for x in timestamps2][::step]
    axs[1].set_xticks(ticks=t2[::step])
    axs[1].set_xticklabels(labels=xticks, fontsize=18)
    yticks = np.arange(-HEIGHT2, max(v2), 500)
    axs[1].set_yticks(ticks=yticks)
    axs[1].set_yticklabels(labels=yticks, fontsize=18)

    axs[0].set_xlim(t1[0], t1[-1])
    axs[0].set_xlabel('t [hh:mm]', fontsize=20)
    axs[0].set_ylabel('P [kW]', fontsize=20)

    axs[1].set_xlim(t1[0], t1[-1])
    axs[1].set_xlabel('t [hh:mm]', fontsize=20)
    axs[1].set_ylabel('w [rpwn]', fontsize=20)

    axs[0].legend(fontsize=20)

    axs[1].legend(fontsize=20)

    fig.savefig(SAVE_PATH + '{}.pdf'.format(title))

    del fig, axs
