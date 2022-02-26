from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from it.polimi.hri_learn.domain.lshafeatures import Trace, TimedTrace, RealValuedVar, FlowCondition, ProbDistribution
from it.polimi.hri_learn.domain.sigfeatures import ChangePoint, Event, SampledSignal, Timestamp, SignalPoint


class SystemUnderLearning:
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

    def __init__(self, rv_vars: List[RealValuedVar], events: List[Event], parse_f, label_f, param_f, **args):
        #
        self.vars = rv_vars
        self.flows = [v.flows for v in rv_vars]
        self.events = events
        self.symbols = {e.symbol: e.label for e in events}
        self.parse_f = parse_f
        self.label_f = label_f
        self.param_f = param_f
        #
        self.signals: List[List[SampledSignal]] = []
        self.timed_traces: List[TimedTrace] = []
        self.traces: List[Trace] = []

        self.name = args['args']['name']
        self.driver = args['args']['driver']
        self.default_m = args['args']['default_m']
        self.default_d = args['args']['default_d']

    def add_distribution(self, d: ProbDistribution, f: FlowCondition):
        self.vars[0].distr.append(d)
        self.vars[0].model2distr[f.f_id].append(d.d_id)

    #
    # TRACE PROCESSING METHODS
    #
    def process_data(self, path: str):
        new_signals: List[SampledSignal] = self.parse_f(path)
        self.signals.append(new_signals)

        driver_sig = [sig for sig in new_signals if sig.label == self.driver][0]

        chg_pts = SystemUnderLearning.find_chg_pts(driver_sig)
        events = [self.label_f(self.events, new_signals, pt.t) for pt in chg_pts]
        new_tt = TimedTrace([pt.t for pt in chg_pts], events)
        self.timed_traces.append(new_tt)
        self.traces.append(Trace(tt=new_tt))

    def get_ht_params(self, segment: List[SignalPoint], flow: FlowCondition):
        return self.param_f(segment, flow)

    def get_segments(self, word: Trace):
        traces: List[int] = [i for i, t in enumerate(self.traces) if str(t).startswith(str(word))]
        if len(traces) == 0:
            return []

        segments = []
        # for all traces, get signal segment from last(word) to the following event
        for trace in traces:
            main_sig_index = [i for i, s in enumerate(self.signals[0]) if s.label == self.vars[0].label][0]
            main_sig = self.signals[trace][main_sig_index]

            if word != '':
                start_timestamp = self.timed_traces[trace].t[max(len(word) - 1, 0)].to_secs()
            else:
                start_timestamp = Timestamp(0, 0, 0, 0, 0, 0).to_secs()

            if len(word) < len(self.timed_traces[trace]):
                end_timestamp = self.timed_traces[trace].t[len(word)].to_secs()
            else:
                end_timestamp = main_sig.points[-1].timestamp.to_secs()

            segment = [pt for pt in main_sig.points if start_timestamp <= pt.timestamp.to_secs() <= end_timestamp]
            segments.append(segment)
        else:
            return segments

    #
    # VISUALIZATION METHODS
    #
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
        for flow in self.flows[0]:
            plt.figure()
            plt.title("Distributions for {}".format(flow.label))
            related_distributions = self.vars[0].get_distr_for_flow(flow.f_id)
            for d in related_distributions:
                distr: Dict[str, float] = d.params
                mu: float = distr['avg']
                sigma: float = distr['var']
                x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
                plt.plot(x, stats.norm.pdf(x, mu, sigma), label='{}({:.6f}, {:.6f})'.format(d, mu, sigma))
            plt.legend()
            plt.show()
