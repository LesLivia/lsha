import configparser
import math
from functools import reduce
from typing import Tuple, List

import numpy as np
import scipy.special as sci
import scipy.stats as stats
from scipy.stats.stats import KstestResult
from tqdm import tqdm

from it.polimi.hri_learn.domain.lshafeatures import TimedTrace, FlowCondition, ProbDistribution, NormalDistribution
from it.polimi.hri_learn.domain.obstable import ObsTable
from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning
from it.polimi.hri_learn.lstar_sha.evt_id import DRIVER_SIG
from it.polimi.hri_learn.lstar_sha.logger import Logger
from it.polimi.hri_learn.lstar_sha.trace_gen import TraceGenerator

LOGGER = Logger('TEACHER')
TG = TraceGenerator()

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()


class Teacher:
    def __init__(self, sul: SystemUnderLearning):
        self.sul = sul

        # System-Dependent Attributes
        self.symbols = sul.symbols
        self.flows = sul.flows
        self.distributions = [v.distr for v in sul.vars]

        # Trace-Dependent Attributes
        self.timed_traces: List[TimedTrace] = sul.timed_traces
        self.signals: List[List[SampledSignal]] = sul.signals

    def add_distribution(self, d: ProbDistribution, f: FlowCondition):
        self.sul.add_distribution(d, f)

    # QUERIES
    @staticmethod
    def derivative(t: List[Timestamp], values: List[float]):
        # returns point-to-point increments for a given time-series
        # (derivative approximation)
        t = [x.to_secs() for x in t]
        increments = []
        try:
            increments = [(v - values[i - 1]) / (t[i] - t[i - 1]) for (i, v) in enumerate(values) if i > 0]
        except ZeroDivisionError:
            avg_dt = sum([x - t[i - 1] for (i, x) in enumerate(t) if i > 0]) / (len(t) - 1)
            increments = [(v - values[i - 1]) / avg_dt for (i, v) in enumerate(values) if i > 0]
        finally:
            return increments

    #############################################
    # MODEL IDENTIFICATION QUERY:
    # for a given prefix (word), gets all corresponding segments
    # and returns the flow condition that best fits such segments
    # If not enough data are available to draw a conclusion, returns None
    #############################################
    def mi_query(self, word: str):
        if word == '':
            return self.flows[0][self.sul.default_m]
        else:
            segments = self.sul.get_segments(word)
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
                    for flow in self.flows[0]:
                        ideal_model = flow.f(interval, segment[0].value)
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
                            best_fit = flow
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
    def ht_query(self, word: str, flow: FlowCondition, save=True):
        if flow is None:
            return None

        if word == '':
            return self.distributions[self.sul.default_d]
        else:
            segments = self.sul.get_segments(word)
            if len(segments) > 0:
                # distr associated with selected flow
                eligible_distributions = self.sul.vars[0].get_distr_for_flow(flow.f_id)

                # randomly distributed metrics for each segment
                metrics = [self.sul.get_ht_params(segment, flow) for segment in segments]
                metrics = [met for met in metrics if met is not None]
                avg_metrics = sum(metrics) / len(metrics)

                # statistical parameters
                alpha, m, max_scs, D_min, best_fit = (0.1, len(metrics), 0, 1000, None)

                # for each eligible distribution, determine if metrics value are a likely sample set
                for (i, distr) in enumerate(eligible_distributions):
                    scs = 0
                    for i in range(100):
                        y = list(np.random.normal(distr.params['avg'], distr.params['var'], m))
                        res: KstestResult = stats.ks_2samp(metrics, y)
                        # counts number of successes with 100 different d populations
                        if res.pvalue > alpha:
                            scs += 1
                    # if d has the best statistic, d becomes the best fit
                    if abs(avg_metrics - distr.params['avg']) < D_min and scs > 0:
                        best_fit = distr
                        D_min = abs(avg_metrics - distr.params['avg'])
                        max_scs = scs

                if max_scs > 0:
                    LOGGER.debug("Accepting N_{} with Y: {}".format(best_fit, max_scs))
                    return best_fit
                else:
                    # rejects H0
                    # if no distribution passes the hyp. test, a new one is created
                    for distr in eligible_distributions:
                        old_avg: float = distr.params['avg']
                        if abs(avg_metrics - old_avg) < old_avg / 5:
                            return distr
                    else:
                        var_metrics = sum([(m - avg_metrics) ** 2 for m in metrics]) / len(metrics)
                        std_dev_metrics = math.sqrt(var_metrics) if var_metrics != 0 else avg_metrics / 10
                        new_distr = NormalDistribution(len(self.distributions[0]), avg_metrics, std_dev_metrics)
                        if save:
                            self.add_distribution(new_distr, flow)
                        return new_distr
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
        n_resample = int(config['LSHA PARAMETERS']['N_min'])
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
        if len(self.get_signals()) >= 2000:
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
