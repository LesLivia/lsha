from typing import List

from it.polimi.hri_learn.domain.sigfeatures import SignalPoint, Timestamp, SampledSignal


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    f = open(path, 'r')
    variables = ['t.ON', 'T_r', 'r.open']
    lines = f.readlines()
    split_indexes = [lines.index('# ' + k + ' #1\n') for k in variables]
    split_lines = [lines[i + 1:split_indexes[ind + 1]] for (ind, i) in enumerate(split_indexes) if
                   i != split_indexes[-1]]
    split_lines.append(lines[split_indexes[-1] + 1:len(lines)])
    new_signals: List[SampledSignal] = []
    for (i, v) in enumerate(variables):
        entries = [line.split(' ') for line in split_lines[i]][1:]
        t = [float(x[0]) for x in entries]
        values = [float(x[1]) for x in entries]
        ts = [Timestamp(0, 0, 0, 0, 0, i) for i in t]
        signal_pts: List[SignalPoint] = [SignalPoint(ts[i], values[i]) for i in range(len(ts))]
        new_signal: SampledSignal = SampledSignal(signal_pts, v)
        new_signals.append(new_signal)
    return new_signals
