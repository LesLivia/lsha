import math
from typing import List

import numpy as np
import statsmodels.tsa.ar_model as ar

from domain.sigfeatures import SignalPoint, ChangePoint, TimeInterval, Labels, SignalType, Position


def pt_dist(p1: Position, p2: Position):
    return math.sqrt(math.fabs(p1.x - p2.x) ** 2 + math.fabs(p1.y - p2.y) ** 2)


def val_factory(x: str, sig_type: SignalType):
    if sig_type == SignalType.NUMERIC:
        return float(x)
    elif sig_type == SignalType.POSITION:
        return Position.parse_pos(x)
    else:
        raise TypeError


def read_signal(lines: List[str], sig_type: SignalType, diff_id: bool = True):
    sig: List[SignalPoint] = []
    for line in lines:
        fields = line.split(':')
        timestamp = float(fields[0])
        if diff_id:
            val = val_factory(fields[2], sig_type)
            sig.append(SignalPoint(timestamp, int(fields[1].replace('hum', '')), val))
        else:
            val = val_factory(fields[1], sig_type)
            sig.append(SignalPoint(timestamp, 1, val))
    return sig


def print_signal(sig: List[SignalPoint]):
    for pt in sig:
        print(pt)


def identify_change_pts(entries: List[SignalPoint]):
    change_pts: List[ChangePoint] = []
    increasing = False

    for (index, entry) in enumerate(entries):
        old_increasing = increasing
        increasing = entry.value > entries[index - 1].value
        if old_increasing != increasing:
            label = Labels.STARTED if increasing else Labels.STOPPED
            change_pts.append(ChangePoint(TimeInterval(entries[index - 1].timestamp, entry.timestamp), label))

    return change_pts


def n_predictions(sig: List[SignalPoint], dt: TimeInterval, n: int, order=1, show_formula=False):
    y_full = list(filter(lambda pt: dt.t_min <= pt.timestamp <= dt.t_max, sig))
    y = list(map(lambda pt: pt.value, y_full))
    model: ar.AutoReg = ar.AutoReg(y, lags=order)
    res = model.fit()
    if show_formula:
        model_formula = 'x(t) = '
        for (index, i) in enumerate(res.params):
            model_formula += '{:.4f}'.format(i)
            model_formula += '*x(t-{})'.format(index) if index > 0 else ''
            model_formula += ' + ' if index < len(res.params) - 1 else ''
        print(model_formula)
    forecasts = res.forecast(n)
    x_fore = np.arange(dt.t_max, dt.t_max + n)
    return res.params, x_fore, forecasts
