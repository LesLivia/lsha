import math
from functools import reduce
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sci
import scipy.stats as stats
from tqdm import tqdm

from it.polimi.hri_learn.domain.sigfeatures import SignalPoint
from it.polimi.hri_learn.lstar_sha.evt_id import EventFactory, DEFAULT_DISTR, DEFAULT_MODEL, MAIN_SIGNAL, MODEL_TO_DISTR_MAP, \
    DRIVER_SIG
from it.polimi.hri_learn.lstar_sha.learner import ObsTable
from it.polimi.hri_learn.lstar_sha.logger import Logger
from it.polimi.hri_learn.lstar_sha.trace_gen import TraceGenerator, CS

LOGGER = Logger()
TG = TraceGenerator()


class Teacher:
    def __init__(self, models, distributions: List[Tuple]):
        # System-Dependent Attributes
        self.symbols = None
        self.models = models
        self.distributions = distributions
        self.evt_factory = EventFactory(None, None, None)

        # Trace-Dependent Attributes
        self.chg_pts: List[List[float]] = []
        self.events = []
        self.signals: List[List[List[SignalPoint]]] = []

    def reset(self):
        self.signals.append([])
        self.evt_factory.clear()

    # SYMBOLS
    def set_symbols(self, symbols):
        self.symbols = symbols
        self.evt_factory.set_symbols(symbols)

    def get_symbols(self):
        return self.symbols

    def compute_symbols(self, guards: List[str], syncs: List[str]):
        symbols = {}
        self.evt_factory.set_guards(guards)
        self.evt_factory.set_channels(syncs)

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
                identifier = ''
                if index > 9:
                    identifier = chr(index + 87)
                else:
                    identifier = str(index)
                symbols[chn + '_' + identifier] = g + ' and ' + chn

        self.set_symbols(symbols)
        self.evt_factory.set_symbols(symbols)

    # CHANGE POINTS
    def set_chg_pts(self, chg_pts: List[float]):
        self.chg_pts.append(chg_pts)

    def get_chg_pts(self):
        return self.chg_pts

    def find_chg_pts(self, timestamps: List[float], values: List[float]):
        chg_pts: List[float] = []

        # IDENTIFY CHANGE PTS IN DRIVER OVERLAY
        prev = values[0]
        for i in range(1, len(values)):
            curr = values[i]
            if curr != prev:
                chg_pts.append(timestamps[i])
            prev = curr

        self.set_chg_pts(chg_pts)

    # SIGNALS
    def get_signals(self):
        return self.signals

    def add_signal(self, signal: List[SignalPoint], trace: int):
        self.signals[trace].append(signal)
        self.evt_factory.add_signal(signal, trace)

    # EVENTS
    def set_events(self, events):
        self.events.append(events)

    def get_events(self):
        return self.events

    def identify_events(self, trace):
        events = {}

        for pt in self.get_chg_pts()[trace]:
            events[pt] = self.evt_factory.label_event(pt, trace)

        self.set_events(events)

    def plot_trace(self, trace: int, title=None, xlabel=None, ylabel=None):
        plt.figure(figsize=(10, 5))

        if title is not None:
            plt.title(title, fontsize=18)
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=18)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=18)

        t = [x.timestamp for x in self.get_signals()[trace][0]]
        v = [x.value for x in self.get_signals()[trace][0]]

        plt.xlim(min(t) - 5, max(t) + 5)
        plt.ylim(0, max(v) + .05)
        plt.plot(t, v, 'k', linewidth=.5)

        x = list(self.events[trace].keys())
        plt.vlines(x, [0] * len(x), [max(v)] * len(x), 'b', '--')
        for (index, e) in enumerate(self.get_events()[trace].values()):
            plt.text(x[index] - 7, max(v) + .01, e, fontsize=18, color='blue')

        plt.show()

    # QUERIES
    def set_models(self, models):
        self.models = models

    def get_models(self):
        return self.models

    def set_distributions(self, distributions):
        self.distributions = distributions

    def get_distributions(self):
        return self.distributions

    def plot_distributions(self):
        for (i_m, m) in enumerate(self.get_models()):
            plt.figure()
            plt.title("Distributions for f_{}".format(i_m))
            related_distributions = list(filter(lambda k: MODEL_TO_DISTR_MAP[k] == i_m, MODEL_TO_DISTR_MAP.keys()))
            for d in related_distributions:
                distr: Tuple = self.get_distributions()[d]
                mu: float = distr[0]
                sigma: float = distr[1]
                x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
                plt.plot(x, stats.norm.pdf(x, mu, sigma), label='N_{}({:.6f}, {:.6f})'.format(d, mu, sigma))
            plt.legend()
            plt.show()

    def get_segments(self, word: str):
        trace_events: List[str] = []
        # converts all trace dicts ({time: evt}) to strings (e_1e_2...)
        for trace in range(len(self.get_events())):
            if len(list(self.get_events()[trace].values())) > 0:
                trace_events.append(reduce(lambda x, y: x + y, list(self.get_events()[trace].values())))
            else:
                trace_events.append('')
        traces = []
        for (i, event_str) in enumerate(trace_events):
            if event_str.startswith(word):
                traces.append(i)
        if len(traces) == 0:
            return []

        segments = []
        # for all traces, get signal segment from last(word) to the following event
        for trace in traces:
            main_sig = self.get_signals()[trace][MAIN_SIGNAL]
            events_in_word = []
            for i in range(0, len(word), 3):
                events_in_word.append(word[i:i + 3])

            if word != '':
                start_timestamp = list(self.get_events()[trace].keys())[max(len(events_in_word) - 1, 0)]
            else:
                start_timestamp = 0
            if len(events_in_word) < len(self.get_events()[trace]):
                end_timestamp = list(self.get_events()[trace].keys())[len(events_in_word)]
            else:
                end_timestamp = main_sig[-1].timestamp

            segment = list(filter(lambda pt: start_timestamp <= pt.timestamp <= end_timestamp, main_sig))
            segments.append(segment)
        else:
            return segments

    @staticmethod
    def derivative(t: List[float], values: List[float]):
        # returns point-to-point increments for a given time-series
        # (derivative approximation)
        increments = []
        try:
            increments = [(v - values[i - 1]) / (t[i] - t[i - 1]) for (i, v) in enumerate(values) if i > 0]
        except ZeroDivisionError:
            avg_dt = sum([x - t[i - 1] for (i, x) in enumerate(t) if i > 0]) / (len(t) - 1)
            increments = [(v - values[i - 1]) / avg_dt for (i, v) in enumerate(values) if i > 0]
        finally:
            return increments

    #############################################
    # MODEL FITTING QUERY:
    # for a given prefix (word), gets all corresponding segments
    # and returns the flow condition that best fits such segments
    # If not enough data are available to draw a conclusion, returns None
    #############################################
    def mi_query(self, word: str):
        if word == '':
            return DEFAULT_MODEL
        else:
            segments = self.get_segments(word)
            if len(segments) > 0:
                fits = []
                for segment in segments:
                    if len(segment) < 10:
                        continue
                    interval = [pt.timestamp for pt in segment]
                    # observed values and (approximate) derivative
                    real_behavior = [pt.value for pt in segment]
                    real_der = self.derivative(interval, real_behavior)
                    min_distance = 10000
                    min_der_distance = 10000
                    best_fit = None

                    # for each model from the given input set
                    for (m_i, model) in enumerate(self.get_models()):
                        ideal_model = model(interval, segment[0].value)
                        distances = [abs(i - real_behavior[index]) for (index, i) in enumerate(ideal_model)]
                        avg_distance = sum(distances) / len(distances)

                        ideal_der = self.derivative(interval, ideal_model)
                        der_distances = [abs(i - real_der[index]) for (index, i) in enumerate(ideal_der)]
                        avg_der_distance = sum(der_distances) / len(der_distances)

                        dist_is_closer = avg_distance < min_distance
                        der_is_closer = avg_der_distance < min_der_distance
                        der_same_sign = sum([v * ideal_der[i] for (i, v) in enumerate(real_der)]) / len(real_der) > 0

                        # compares the observed behavior with the ideal one (values and derivatives)
                        if dist_is_closer and der_is_closer and der_same_sign:
                            min_distance = avg_distance
                            min_der_distance = avg_der_distance
                            best_fit = m_i
                    else:
                        fits.append(best_fit)

                unique_fits = set(fits)
                freq = -1
                best_fit = None
                for f in unique_fits:
                    matches = sum([x == f for x in fits]) / len(fits)
                    if matches > freq:
                        freq = matches
                        best_fit = f
                if freq > 0.75:
                    return best_fit
                else:
                    LOGGER.info("!! INCONSISTENT PHYSICAL BEHAVIOR !!")
                    return None
            else:
                return None

    @staticmethod
    def get_theta_th(P_0: float, N: int, alpha: float = 0.05):
        # returns the maximum number of failures allowed by conf. level alpha
        for theta in range(0, N, 1):
            alpha_th = 0
            for K in range(theta, N, 1):
                binom = sci.binom(N, K)
                alpha_th += binom * (P_0 ** K) * ((1 - P_0) ** (N - K))
            if alpha_th <= alpha:
                return theta
        return None

    #############################################
    # HYPOTHESIS TESTING QUERY:
    # for a given prefix (word), gets all corresponding segments
    # and returns the random variable that best fits the randomly
    # generated model parameters.
    # If none of the available rand. variables fits the set of segments,
    # a new one is added
    # If available data are not enough to draw a conclusion, returns None
    #############################################
    def ht_query(self, word: str, model=DEFAULT_MODEL, save=True):
        if model is None:
            return None

        if word == '':
            return DEFAULT_DISTR
        else:
            segments = self.get_segments(word)
            if len(segments) > 0:
                eligible_distributions = [k for k in MODEL_TO_DISTR_MAP.keys() if
                                          MODEL_TO_DISTR_MAP[k] == model]

                metrics = [self.evt_factory.get_ht_metric(segment, model) for segment in segments]
                metrics = [met for met in metrics if met is not None]
                alpha = 0.1
                m = len(metrics)
                max_scs = 0
                D_min = 1000
                best_fit = None
                avg_metrics = sum(metrics) / len(metrics)
                for (i, d) in enumerate(eligible_distributions):
                    distr: Tuple[float, float, float] = self.get_distributions()[d]
                    scs = 0
                    for i in range(100):
                        y = list(np.random.normal(distr[0], distr[1], m))
                        res = stats.ks_2samp(metrics, y)
                        if res.pvalue > alpha:
                            scs += 1
                    if abs(avg_metrics - distr[0]) < D_min and scs > 0:
                        best_fit = d
                        D_min = abs(avg_metrics - distr[0])
                        max_scs = scs

                if max_scs > 0:
                    LOGGER.debug("Accepting N_{} with Y: {}".format(best_fit, max_scs))
                    return best_fit
                else:
                    # rejects H0
                    # if no distribution passes the hyp. test, a new one is created
                    for d in eligible_distributions:
                        distr: Tuple[float, float, float] = self.get_distributions()[d]
                        old_avg: float = distr[0]
                        if abs(avg_metrics - old_avg) < old_avg / 10:
                            return d
                    # FIXME
                    if len(self.get_distributions()) >= 3:
                        return None
                    if save:
                        var_metrics = sum([(m - avg_metrics) ** 2 for m in metrics]) / len(metrics)
                        std_dev_metrics = math.sqrt(var_metrics) if var_metrics != 0 else avg_metrics / 10
                        self.get_distributions().append((avg_metrics, std_dev_metrics, len(metrics)))
                        # and added to the map of eligible distr. for the selected model
                        new_distr_index = len(self.get_distributions()) - 1
                        MODEL_TO_DISTR_MAP[new_distr_index] = model
                    else:
                        new_distr_index = len(self.get_distributions()) + 1

                    return new_distr_index
            else:
                return None

    #############################################
    # ROW EQUALITY QUERY:
    # checks if two rows (row(s1), row(s2)) are weakly equal
    # returns true/false
    #############################################
    def eqr_query(self, s1: str, s2: str, row1: List[Tuple], row2: List[Tuple], strict=False):
        if strict:
            return row1 == row2

        for (c_i, cell) in enumerate(row1):
            cell_is_filled = cell[0] is not None and cell[1] is not None
            cell2_is_filled = row2[c_i][0] is not None and row2[c_i][1] is not None
            # if both rows have filled cells which differ from each other,
            # weak equality is violated
            if cell_is_filled and cell2_is_filled and cell != row2[c_i]:
                return False
        return True

    def process_trace(self, path: str):
        prev_traces = len(self.get_signals())
        new_traces = self.evt_factory.parse_traces(path)
        for (t, trace) in enumerate(new_traces):
            self.reset()
            driver_t = []
            driver_v = []
            for (i, signal) in enumerate(trace):
                self.add_signal(signal, t + prev_traces)
                if i == DRIVER_SIG:
                    driver_t = [pt.timestamp for pt in signal]
                    driver_v = [pt.value for pt in signal]
            self.find_chg_pts(driver_t, driver_v)
            self.identify_events(t + prev_traces)

    #############################################
    # KNOWLEDGE REFINEMENT QUERY:
    # checks if there are ambiguous words in the observation table
    # if so, it samples new traces (through the TraceGenerator)
    # to gain more knowledge about the system under learning
    #############################################
    def ref_query(self, table: ObsTable):
        n_resample = 10
        S = table.get_S()
        upp_obs = table.get_upper_observations()
        lS = table.get_low_S()
        low_obs = table.get_lower_observations()

        # find all words which are ambiguous
        # (equivalent to multiple rows)
        amb_words = []
        for (i, row) in enumerate(upp_obs + low_obs):
            s = S[i] if i < len(upp_obs) else lS[i - len(upp_obs)]
            for (e_i, e) in enumerate(table.get_E()):
                if len(self.get_segments(s + e)) < n_resample:
                    amb_words.append(s + e)

            eq_rows = []
            if row[0] == (None, None):
                continue

            for (j, row_2) in enumerate(upp_obs):
                row_2_populated = row_2[0] != (None, None)
                if row_2_populated and i != j and self.eqr_query(s, S[j], row, row_2):
                    eq_rows.append(row_2)
            uq = []
            for eq in eq_rows:
                if eq not in uq:
                    uq.append(eq)

            if len(uq) > 1:
                amb_words.append(s)

        # sample new traces only for ambiguous words which
        # are not prefixes of another ambiguous word
        uq = []
        for (i, w) in enumerate(amb_words):
            is_prefix = False
            for (j, w2) in enumerate(amb_words):
                if i != j and w2.startswith(w):
                    is_prefix = True
            if not is_prefix:
                uq.append(w)

        for word in tqdm(uq, total=len(uq)):
            for e in table.get_E():
                TG.set_word(word + e)
                path = TG.get_traces()
                if path is not None:
                    for sim in path:
                        self.process_trace(sim)
                else:
                    LOGGER.debug('!! An error occurred while generating traces !!')

    #############################################
    # COUNTEREXAMPLE QUERY:
    # looks for a counterexample to current obs. table
    # returns counterexample t if:
    # -> t highlights non-closedness
    # -> t highlights non-consistency
    #############################################
    def get_counterexample(self, table: ObsTable):
        # FIXME
        if len(self.get_signals()) >= 200:
            return None

        S = table.get_S()
        low_S = table.get_low_S()

        trace_events: List[str] = []
        for trace in range(len(self.get_events())):
            if len(list(self.get_events()[trace].values())) > 0:
                trace_events.append(reduce(lambda x, y: x + y, list(self.get_events()[trace].values())))
            else:
                trace_events.append('')
        max_events = int(max([len(t) for t in trace_events]))

        not_counter = []
        for (i, event_str) in tqdm(enumerate(trace_events), total=len(trace_events)):
            for j in range(3, max_events + 1, 3):
                # event_str[:j] not in S and event_str[:j] not in low_S and
                if event_str[:j] not in S and event_str[:j] not in low_S and event_str[:j] not in not_counter:
                    # fills hypothetical new row
                    new_row = []
                    for (e_i, e_word) in enumerate(table.get_E()):
                        word = event_str[:j] + e_word
                        id_model = self.mi_query(word)
                        id_distr = self.ht_query(word, id_model, save=False)
                        if id_model is not None and id_distr is not None:
                            new_row.append((id_model, id_distr))
                        else:
                            new_row.append((None, None))
                    new_row_is_filled = any([t[0] is not None and t[1] is not None for t in new_row])
                    # if there are sufficient data to fill the new row
                    if new_row_is_filled:
                        new_row_is_present = False
                        eq_rows = []
                        for (s_i, s_word) in enumerate(S):
                            row = table.get_upper_observations()[s_i]
                            # checks if there are weakly equal rows (-> row is present)
                            if self.eqr_query(event_str[:j], s_word, new_row, row):
                                new_row_is_present = True
                                eq_rows.append(row)
                        uq = []
                        for e in eq_rows:
                            # checks if new row would be ambiguous (-> not ambiguous)
                            if e not in uq:
                                uq.append(e)
                        not_ambiguous = len(uq) <= 1

                        if new_row and not new_row_is_present:
                            # found non-closedness
                            LOGGER.warn("!! MISSED NON-CLOSEDNESS !!")
                            return event_str[:j]
                        elif not_ambiguous:
                            # checks non-consistency only for rows that are not ambiguous
                            for (s_i, s_word) in enumerate(S):
                                old_row = table.get_upper_observations()[s_i] if s_i < len(S) else \
                                    table.get_lower_observations()[s_i - len(S)]
                                # finds weakly equal rows in S
                                if self.eqr_query(s_word, event_str[:j], old_row, new_row):
                                    for a in self.get_symbols():
                                        # if the hypothetical discrimating event is already in E
                                        discr_is_prefix = False
                                        for e in table.get_E():
                                            if e.startswith(a):
                                                continue
                                        # else checks all 1-step distant rows
                                        if s_word + a in S:
                                            old_row_a = table.get_upper_observations()[S.index(s_word + a)]
                                        elif s_word + a in low_S:
                                            old_row_a = table.get_lower_observations()[low_S.index(s_word + a)]
                                        else:
                                            continue
                                        row_1_filled = old_row_a[0] != (None, None)
                                        row_2 = []
                                        for e in table.get_E():
                                            id_model_2 = self.mi_query(event_str[:j] + a + e)
                                            id_distr_2 = self.ht_query(event_str[:j] + a + e, id_model_2, save=False)
                                            if id_model_2 is None or id_distr_2 is None:
                                                row_2.append((None, None))
                                            else:
                                                row_2.append((id_model_2, id_distr_2))
                                        row_2_filled = row_2[0] != (None, None)
                                        if row_1_filled and row_2_filled and not discr_is_prefix and \
                                                not self.eqr_query(event_str[:j] + a, s_word + a, row_2, old_row_a):
                                            LOGGER.warn("!! MISSED NON-CONSISTENCY ({}, {}) !!".format(a, s_word))
                                            return event_str[:j]
                            else:
                                not_counter.append(event_str[:j])
        else:
            return None
