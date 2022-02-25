from it.polimi.hri_learn.domain.sigfeatures import SignalPoint, Timestamp


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    f = open(path, 'r')
    variables = ['t.ON', 'T_r', 'r.open']
    lines = f.readlines()
    split_indexes = [lines.index(k + ':\n') for k in variables]
    split_lines = [lines[i + 1:split_indexes[ind + 1]] for (ind, i) in enumerate(split_indexes) if
                   i != split_indexes[-1]]
    split_lines.append(lines[split_indexes[-1] + 1:len(lines)])
    traces = len(split_lines[0])
    new_traces = []
    for trace in range(traces):
        new_traces.append([])
        for (i, v) in enumerate(variables):
            entries = split_lines[i][trace].split(' ')
            entries = entries[1:]
            for e in entries:
                new = e.replace('(', '')
                new = new.replace(')', '')
                entries[entries.index(e)] = new
            t = [float(x.split(',')[0]) for x in entries]
            v = [float(x.split(',')[1]) for x in entries]
            ts = [Timestamp(0, 0, 0, 0, 0, i) for i in t]
            signal = [SignalPoint(ts[i], v[i]) for i in range(len(ts))]
            new_traces[-1].append(signal)
    return new_traces
