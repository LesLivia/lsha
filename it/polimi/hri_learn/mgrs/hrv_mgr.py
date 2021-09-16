from typing import List

import biosppy
import matplotlib.pyplot as plt
import numpy as np
from pyhrv.time_domain import time_domain
from pyhrv.tools import nn_intervals

from domain.sigfeatures import SignalPoint


def get_hrv_data(data: List[SignalPoint], sub_id: str = '0', phase: str = 'Base', show=False):
    ecg = list(map(lambda d: d.value, data))
    t = list(map(lambda d: d.timestamp, data))

    peaks = biosppy.ecg.ecg(np.array(ecg), show=False)[2]

    if show:
        plt.figure(figsize=(50, 10))
        plt.title('Subject ' + sub_id + ' ECG and R-peaks during ' + phase, fontsize=30, fontweight='bold')
        plt.xlabel('t [ms]', fontsize=25, fontweight='bold')
        plt.ylabel('ECG [mV]', fontsize=25, fontweight='bold')
        plt.xticks(np.arange(t[0], t[len(t) - 1], step=100), fontsize=25)
        plt.yticks(np.arange(min(ecg), max(ecg), step=0.25), fontsize=25)
        plt.plot(t, ecg, linewidth=1.0, color='darkturquoise')
        for peak in peaks:
            plt.plot(t[peak], ecg[peak], color='tomato', marker='o', markersize=18)
        # plt.savefig(FIG_PATH + '/ecg/' + phase + '/sub' + sub_id + '.pdf')
        plt.show()

    nni = nn_intervals([t[peak] for peak in peaks])

    res = time_domain(nni=nni, rpeaks=peaks, signal=np.array(ecg), plot=False, show=False)
    return peaks, res
