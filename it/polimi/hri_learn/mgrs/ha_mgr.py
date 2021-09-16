from typing import List

from domain.hafeatures import Edge, Location, LOCATIONS
from domain.sigfeatures import ChangePoint, Labels, SignalPoint


def guard_factory(pt: ChangePoint, guard_key: str, signal: List[SignalPoint] = None):
    if signal is None:
        return '{} <= {} <= {}'.format(pt.dt.t_min, guard_key, pt.dt.t_max)
    else:
        signal_pts = []
        for sig_pt in signal:
            if pt.dt.t_min <= sig_pt.timestamp <= pt.dt.t_max:
                signal_pts.append(sig_pt.value)
        return '{:.2f} <= {} <= {:.2f}'.format(min(signal_pts), guard_key, max(signal_pts))


def identify_edges(chg_pts: List[ChangePoint], guard_key: str = 't', signal: List[SignalPoint] = None):
    edges: List[Edge] = []

    for pt in chg_pts:
        start: Location = LOCATIONS[0] if pt.event == Labels.STARTED else LOCATIONS[1]
        dest: Location = LOCATIONS[1] if pt.event == Labels.STARTED else LOCATIONS[0]
        guard = guard_factory(pt, guard_key, signal)
        edges.append(Edge(start, dest, guard))

    return edges
