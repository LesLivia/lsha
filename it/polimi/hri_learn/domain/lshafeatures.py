from typing import List, Dict, Tuple
from it.polimi.hri_learn.domain.sigfeatures import ChangePoint, Event

LOCATION_FORMATTER = 'q_{}'
EMPTY_STRING = '\u03B5'


class ProbDistribution:
    DISTR_FORMATTER = 'D_{}'

    def __init__(self, d_id: int, params: Dict[str, float]):
        self.params = params
        self.d_id = d_id
        self.label = self.DISTR_FORMATTER.format(d_id)
        self.d_id = d_id

    def __eq__(self, other):
        for p in self.params:
            if p not in other.params or self.params[p] != other.params[p]:
                return False
        return True

    def __str__(self):
        return self.label if self.params is not None else EMPTY_STRING


class NormalDistribution(ProbDistribution):
    DISTR_FORMATTER = 'N_{}'

    def __init__(self, d_id: int, avg: float, var: float):
        params = {'avg': avg, 'var': var}
        super().__init__(d_id, params)


class FlowCondition:
    MODEL_FORMATTER = 'f_{}'

    def __init__(self, f_id: int, f):
        self.f = f
        self.f_id = f_id
        self.label = self.MODEL_FORMATTER.format(f_id)

    def __eq__(self, other):
        return self.f == other.f

    def __str__(self):
        return self.label if self.f is not None else EMPTY_STRING


class RealValuedVar:
    def __init__(self, flows: List[FlowCondition], distr: List[ProbDistribution], label: str = None):
        self.flows = flows
        self.distr = distr
        self.label = label

    def __eq__(self, other):
        same_flows = all([f in other.flows for f in self.flows])
        same_distr = all([d in other.distr for d in self.distr])
        return same_flows and same_distr


class TimedTrace:
    def __init__(self, chg_pts: List[ChangePoint]):
        self.chg_pts = chg_pts

    def __eq__(self, other):
        return all([p in other.chg_pts for p in self.chg_pts])

    def __len__(self):
        return len(self.chg_pts)


class Trace:
    def __init__(self, tt: TimedTrace):
        self.events = [x.event for x in tt.chg_pts]

    def __str__(self):
        return ','.join([str(e) for e in self.events])

    def __len__(self):
        return len(self.events)


class State:
    def __init__(self, vars: List[Tuple[FlowCondition, ProbDistribution]]):
        self.vars = vars
        self.label = ';'.join(['(' + str(pair[0]) + ',' + str(pair[1]) + ')' for pair in vars])

    def __str__(self):
        return self.label

    def observed(self):
        return any([pair[0].f is not None and pair[1] is not None for pair in self.vars])


class SystemUnderLearning:
    def __init__(self, name: str, vars: List[RealValuedVar], events: List[Event], parse_f, label_f):
        self.name = name
        self.vars = vars
        self.flows = [v.flows for v in vars]
        self.events = events
        self.symbols = {}
        self.parse_f = parse_f
        self.label_f = label_f

    def compute_symbols(self):
        symbols = {}
        guards = [e.guard for e in self.events if len(e.guard) > 1]
        syncs = [e.chan for e in self.events]

        # Compute all guards combinations
        guards_comb = [''] * 2 ** len(guards)
        for (i, g) in enumerate(guards):
            pref = ''
            for j in range(2 ** len(guards)):
                guards_comb[j] += pref + g
                if (j + 1) % ((2 ** len(guards)) / (2 ** (i + 1))) == 0:
                    pref = '!' if pref == '' else ''

        # Combine all guards with channels
        for chn in syncs:
            for (index, g) in enumerate(guards_comb):
                if index > 9:
                    identifier = chr(index + 87)
                else:
                    identifier = str(index)
                symbols[chn + '_' + identifier] = g + ' and ' + chn

        self.symbols = symbols
