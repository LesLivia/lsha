from typing import List, Dict, Tuple

from it.polimi.hri_learn.domain.sigfeatures import ChangePoint, Event, SampledSignal, Timestamp

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
    @staticmethod
    def compute_symbols(events: List[Event]):
        symbols = {}
        guards = [e.guard for e in events if len(e.guard) > 1]
        syncs = [e.chan for e in events]

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

        return symbols

    @staticmethod
    def find_chg_pts(driver: SampledSignal):
        timestamps = [pt.timestamp for pt in driver.points]
        values = [pt.value for pt in driver.points]
        chg_pts: List[ChangePoint] = []

        # IDENTIFY CHANGE PTS IN DRIVER OVERLAY
        prev = values[0]
        for i in range(1, len(values)):
            curr = values[i]
            if curr != prev:
                chg_pts.append(ChangePoint(timestamps[i]))
            prev = curr

        return chg_pts

    def __init__(self, rv_vars: List[RealValuedVar], events: List[Event], parse_f, label_f, **args):
        self.name = args['args']['name']
        self.driver = args['args']['driver']
        self.vars = rv_vars
        self.flows = [v.flows for v in rv_vars]
        self.events = events
        self.symbols = SystemUnderLearning.compute_symbols(events)
        self.parse_f = parse_f
        self.label_f = label_f
        self.signals: List[List[SampledSignal]] = []
        self.timed_traces: List[TimedTrace] = []
        self.traces: List[Trace] = []

    def process_data(self, path: str):
        new_signals: List[SampledSignal] = self.parse_f(path)
        self.signals.append(new_signals)

        driver_sig = [sig for sig in new_signals if sig.label == self.driver][0]

        chg_pts = SystemUnderLearning.find_chg_pts(driver_sig)
        events = [self.label_f(self.events, new_signals, pt.t) for pt in chg_pts]
        new_tt = TimedTrace([pt.t for pt in chg_pts], events)
        self.timed_traces.append(new_tt)
        self.traces.append(Trace(new_tt))
