from typing import List, Dict, Tuple

from sha_learning.domain.sigfeatures import Event, Timestamp

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
        return self.f_id == other.f_id

    def __str__(self):
        return self.label if self.f is not None else EMPTY_STRING

    def __hash__(self):
        return self.f_id


class RealValuedVar:
    def __init__(self, flows: List[FlowCondition], distr: List[ProbDistribution],
                 m2d: Dict[int, List[int]], label: str = None):
        self.flows = flows
        self.distr = distr
        self.model2distr = m2d
        self.label = label

    def __eq__(self, other):
        same_flows = all([f in other.flows for f in self.flows])
        same_distr = all([d in other.distr for d in self.distr])
        return same_flows and same_distr

    def get_distr_for_flow(self, x: int):
        related_distr = self.model2distr[x]
        return list(filter(lambda d: d.d_id in related_distr, self.distr))


class TimedTrace:
    def __init__(self, t: List[Timestamp], e: List[Event]):
        self.t = t
        self.e = e

    def __eq__(self, other):
        return all([ts == other.t[i] and self.e[i] == other.e[i] for i, ts in enumerate(self.t)])

    def __len__(self):
        return len(self.t)


class Trace:
    def __init__(self, events: List[Event] = None, tt: TimedTrace = None):
        if tt is not None:
            self.events = tt.e
        else:
            self.events = events

    def __str__(self):
        if len(self.events) == 0:
            return EMPTY_STRING
        else:
            return ''.join([e.symbol for e in self.events])

    def __eq__(self, other):
        return str(self) == str(other)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        return self.events[item]

    def __add__(self, other):
        return Trace(events=self.events + other.events)

    def sub_prefix(self, prefix):
        return Trace(events=self.events[len(prefix.events):])

    def __hash__(self):
        return hash(str(self))

    def get_prefixes(self):
        prefixes: List[Trace] = []
        for i in range(len(self)):
            if i == 0:
                prefixes.append(Trace([self[0]]))
            else:
                prefixes.append(Trace(self[:i + 1]))
        return prefixes

    def startswith(self, word):
        if len(word) > len(self):
            return False

        for i, e in enumerate(word):
            if self.events[i].symbol != e.symbol:
                return False
        else:
            return True


class State:
    def __init__(self, vars: List[Tuple[FlowCondition, ProbDistribution]]):
        self.vars = vars
        self.label = ''
        for i, v in enumerate(vars):
            if v[0] is not None and v[1] is not None:
                self.label += '({},{})'.format(v[0], v[1])
            else:
                self.label += '({},{})'.format(EMPTY_STRING, EMPTY_STRING)
            if i < len(vars) - 1:
                self.label += ';'

    def __str__(self):
        return self.label

    def __eq__(self, other):
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def observed(self):
        return any([pair[0] is not None and pair[1] is not None for pair in self.vars])
