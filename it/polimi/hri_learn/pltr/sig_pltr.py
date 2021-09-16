from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import mgrs.sig_mgr as sig_mgr
from domain.sigfeatures import SignalPoint, ChangePoint, Labels, TimeInterval


def plot_sig(entries: List[SignalPoint], chg_pts: List[ChangePoint], with_pred=False, n_pred=0):
    plt.figure(figsize=(30, 10))
    plt.xlabel('t [s]', fontsize=24)
    plt.ylabel('F [%]', fontsize=24)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.grid(linestyle="--")
    x = list(map(lambda e: e.timestamp, entries))
    y_max = list(map(lambda e: e.value, entries))
    y_min = [0] * len(y_max)
    plt.plot(x, y_max, 'k.', label='sensor readings')
    plt.vlines(x, y_min, y_max, 'lightgray')

    colors = ['turquoise', 'tomato']
    labels = []
    for (index, pt) in enumerate(chg_pts):
        color = colors[0] if pt.event == Labels.STARTED else colors[1]
        items = list(filter(lambda e: pt.dt.t_min <= e.timestamp <= pt.dt.t_min, entries))
        x = list(map(lambda i: i.timestamp, items))
        y = list(map(lambda i: i.value, items))
        if pt.event not in labels:
            plt.vlines(x, [0] * len(y), y, color=color, label=pt.event)
            labels.append(pt.event)
        else:
            plt.vlines(x, [0] * len(y), y, color=color)

        if with_pred:
            try:
                dt = TimeInterval(pt.dt.t_min, chg_pts[index + 1].dt.t_min)
            except IndexError:
                entries_ts = [entry.timestamp for entry in entries]
                dt = TimeInterval(pt.dt.t_min, max(entries_ts))
            param_est, x_fore, forecasts = sig_mgr.n_predictions(entries, dt, n_pred, order=4, show_formula=True)
            step = entries[len(entries) - 1].timestamp - entries[len(entries) - 2].timestamp
            x_fore = [(ts_fore - x_fore[0]) * step + ts_fore for ts_fore in x_fore]
            plt.plot(x_fore, forecasts, 'b.', label='predictions')
            plt.vlines(x_fore, [0] * n_pred, forecasts, 'blue')
            f = interp1d(x_fore, forecasts, kind='linear', axis=0)
            x_interp = np.arange(x_fore[0], x_fore[-1], step=1)
            y_interp = f(x_interp)
            plt.plot(x_interp, y_interp, 'aquamarine')
    plt.legend(fontsize=15)
    plt.show()
