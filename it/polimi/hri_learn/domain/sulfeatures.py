from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from it.polimi.hri_learn.domain.lshafeatures import Trace, TimedTrace, RealValuedVar
from it.polimi.hri_learn.domain.sigfeatures import ChangePoint, Event, SampledSignal


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
        #
        self.vars = rv_vars
        self.flows = [v.flows for v in rv_vars]
        self.events = events
        self.symbols = SystemUnderLearning.compute_symbols(events)
        self.parse_f = parse_f
        self.label_f = label_f
        #
        self.signals: List[List[SampledSignal]] = []
        self.timed_traces: List[TimedTrace] = []
        self.traces: List[Trace] = []

        self.name = args['args']['name']
        self.driver = args['args']['driver']

    def process_data(self, path: str):
        new_signals: List[SampledSignal] = self.parse_f(path)
        self.signals.append(new_signals)

        driver_sig = [sig for sig in new_signals if sig.label == self.driver][0]

        chg_pts = SystemUnderLearning.find_chg_pts(driver_sig)
        events = [self.label_f(self.events, new_signals, pt.t) for pt in chg_pts]
        new_tt = TimedTrace([pt.t for pt in chg_pts], events)
        self.timed_traces.append(new_tt)
        self.traces.append(Trace(new_tt))

    def plot_trace(self, i=None, title=None, xlabel=None, ylabel=None):
        plt.figure(figsize=(10, 5))

        if title is not None:
            plt.title(title, fontsize=18)
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=18)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=18)

        signal_to_plot = [i for i, s in enumerate(self.signals[0]) if s.label == self.vars[0].label][0]
        to_plot = [self.timed_traces[i]] if i is not None else self.timed_traces
        for i, tt in enumerate(to_plot):
            sig = self.signals[i][signal_to_plot].points
            t = [pt.timestamp.to_secs() for pt in sig]
            v = [pt.value for pt in sig]

            plt.xlim(min(t) - 5, max(t) + 5)
            plt.ylim(0, max(v) + .05)
            plt.plot(t, v, 'k', linewidth=.5)

            plt.vlines([ts.to_secs() for ts in tt.t], [0] * len(tt), [max(v)] * len(tt), 'b', '--')
            for index, evt in enumerate(tt.e):
                plt.text(tt.t[index].to_secs() - 7, max(v) + .01, str(evt), fontsize=18, color='blue')

            plt.show()

    def plot_distributions(self):
        for flow in self.flows:
            plt.figure()
            plt.title("Distributions for {}".format(flow[0].label))
            related_distributions = self.vars[0].get_distr_for_flow(flow[0].f_id)
            for d in related_distributions:
                distr: Dict[str, float] = d.params
                mu: float = distr['avg']
                sigma: float = distr['var']
                x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
                plt.plot(x, stats.norm.pdf(x, mu, sigma), label='N_{}({:.6f}, {:.6f})'.format(d, mu, sigma))
            plt.legend()
            plt.show()
