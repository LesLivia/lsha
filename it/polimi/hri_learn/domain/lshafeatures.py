from typing import List, Dict, Tuple

from it.polimi.hri_learn.domain.sigfeatures import Event, Timestamp

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
    def __init__(self, tt: TimedTrace):
        self.events = tt.e

    def __str__(self):
        return ''.join([e.symbol for e in self.events])

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
