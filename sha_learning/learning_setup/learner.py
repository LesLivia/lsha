import configparser
import os
from typing import List, Tuple, Dict, Set

from sha_learning.domain.lshafeatures import State, FlowCondition, ProbDistribution
from sha_learning.domain.obstable import ObsTable, Row, Trace
from sha_learning.domain.shafeatures import StochasticHybridAutomaton, Location, Edge
from sha_learning.learning_setup.logger import Logger
from sha_learning.learning_setup.teacher import Teacher

config = configparser.ConfigParser()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

LOGGER = Logger('LEARNER')


class Learner:
    def __init__(self, teacher: Teacher, table: ObsTable):
        self.symbols = teacher.symbols
        self.TEACHER = teacher
        default_table = table
        self.obs_table = table if table is not None else default_table

    def fill_row(self, row: Row, i: int, s_word: Trace, obs: List[Row]):
        for j, t_word in enumerate(self.obs_table.get_E()):
            # if cell is yet to be filled,
            # asks teacher to answer queries
            # and fills cell with answers
            if not obs[i].state[j].observed():
                identified_model: FlowCondition = self.TEACHER.mi_query(s_word + t_word)
                identified_distr: ProbDistribution = self.TEACHER.ht_query(s_word + t_word, identified_model)
                if identified_model is None or identified_distr is None:
                    identified_model = None
                    identified_distr = None
                row.state[j] = State([(identified_model, identified_distr)])
                if not State([(identified_model, identified_distr)]).observed() and j == 0:
                    break
        return row

    def fill_table(self):
        upp_obs: List[Row] = self.obs_table.get_upper_observations()
        for i, s_word in enumerate(self.obs_table.get_S()):
            row: Row = Row(upp_obs[i].state.copy())
            row = self.fill_row(row, i, s_word, upp_obs)
            upp_obs[i] = row
        self.obs_table.set_upper_observations(upp_obs)

        low_obs: List[Row] = self.obs_table.get_lower_observations()
        for i, s_word in enumerate(self.obs_table.get_low_S()):
            row: Row = Row(low_obs[i].state.copy())
            row = self.fill_row(row, i, s_word, low_obs)
            low_obs[i] = row
        self.obs_table.set_lower_observations(low_obs)

    def is_closed(self):
        upp_obs: List[Row] = self.obs_table.get_upper_observations()
        low_obs: List[Row] = self.obs_table.get_lower_observations()
        for l_i, row in enumerate(low_obs):
            if not row.is_populated():
                continue
            row_is_in_upper = False
            for s_i, s_word in enumerate(self.obs_table.get_S()):
                if self.TEACHER.eqr_query(row, upp_obs[s_i]):
                    row_is_in_upper = True
                    break
            if not row_is_in_upper:
                return False
        else:
            return True

    def is_consistent(self):
        events = self.TEACHER.sul.events
        upp_obs = self.obs_table.get_upper_observations()
        pairs: List[Tuple[Trace, Trace]] = []
        # FIXME: each pair shows up twice, duplicates should be cleared
        for index, row in enumerate(upp_obs):
            equal_rows = [i for (i, r) in enumerate(upp_obs) if
                          index != i and self.TEACHER.eqr_query(row, r, strict=True)]
            S = self.obs_table.get_S()
            equal_pairs = [(S[index], S[equal_i]) for equal_i in equal_rows]
            pairs += equal_pairs
        if len(pairs) == 0:
            return True, None
        else:
            for pair in pairs:
                for e in events:
                    try:
                        new_pair_1 = self.obs_table.get_S().index(pair[0] + Trace([e]))
                        new_row_1 = self.obs_table.get_upper_observations()[new_pair_1]
                    except ValueError:
                        new_pair_1 = self.obs_table.get_low_S().index(pair[0] + Trace([e]))
                        new_row_1 = self.obs_table.get_lower_observations()[new_pair_1]

                    try:
                        new_pair_2 = self.obs_table.get_S().index(pair[1] + Trace([e]))
                        new_row_2 = self.obs_table.get_upper_observations()[new_pair_2]
                    except ValueError:
                        new_pair_2 = self.obs_table.get_low_S().index(pair[1] + Trace([e]))
                        new_row_2 = self.obs_table.get_lower_observations()[new_pair_2]

                    rows_different = not self.TEACHER.eqr_query(new_row_1, new_row_2)
                    if new_row_1.is_populated() and new_row_2.is_populated() and rows_different:
                        for e_i, e_word in enumerate(self.obs_table.get_E()):
                            if new_row_1.state[e_i] != new_row_2.state[e_i] \
                                    and (Trace([e]) + e_word) not in self.obs_table.get_E():
                                LOGGER.warn('INCONSISTENCY: {}-{}'.format(pair[0] + Trace([e]), pair[1] + Trace([e])))
                                return False, Trace([e]) + e_word
            else:
                return True, None

    def make_closed(self):
        upp_obs: List[Row] = self.obs_table.get_upper_observations()
        low_S: List[Trace] = self.obs_table.get_low_S()
        low_obs: List[Row] = self.obs_table.get_lower_observations()
        for index, row in enumerate(low_obs):
            # if there is a populated row in lower portion that is not in the upper portion
            # the corresponding word is added to the S word set
            row_present = any([self.TEACHER.eqr_query(row, row_2) for row_2 in upp_obs])
            if row.is_populated() and not row_present:
                upp_obs.append(Row(row.state))
                new_s_word: Trace = low_S[index]
                self.obs_table.add_S(new_s_word)
                low_obs.pop(index)
                self.obs_table.del_low_S(index)
                # lower portion is then updated with all combinations of
                # new S word and all possible symbols
                for event in self.TEACHER.sul.events:
                    self.obs_table.add_low_S(new_s_word + Trace([event]))
                    empty_state = State([(None, None)])
                    new_row: Row = Row([empty_state] * len(self.obs_table.get_E()))
                    low_obs.append(new_row)
        self.obs_table.set_upper_observations(upp_obs)
        self.obs_table.set_lower_observations(low_obs)

    def make_consistent(self, discr_sym: Trace):
        self.obs_table.add_E(discr_sym)
        upp_obs: List[Row] = self.obs_table.get_upper_observations()
        low_obs: List[Row] = self.obs_table.get_lower_observations()
        for s_i in range(len(upp_obs)):
            upp_obs[s_i].state.append(State([(None, None)]))
        for s_i in range(len(low_obs)):
            low_obs[s_i].state.append(State([(None, None)]))

    def add_counterexample(self, counterexample: Trace):
        upp_obs = self.obs_table.get_upper_observations()
        low_obs = self.obs_table.get_lower_observations()

        # add counterexample and all its prefixes to S
        for i, e in enumerate(counterexample):
            new_trace = Trace(counterexample[:i + 1])
            if new_trace not in self.obs_table.get_S():
                if new_trace in self.obs_table.get_low_S():
                    to_copy: Row = low_obs[self.obs_table.get_low_S().index(new_trace)]
                else:
                    to_copy: Row = Row([State([(None, None)])] * len(self.obs_table.get_E()))
                self.obs_table.get_S().append(new_trace)
                upp_obs.append(Row(to_copy.state))

            if new_trace in self.obs_table.get_low_S():
                row_index = self.obs_table.get_low_S().index(new_trace)
                self.obs_table.get_lower_observations().pop(row_index)
                self.obs_table.get_low_S().pop(row_index)

            # add 1-step away words to low_S
            for a in self.TEACHER.sul.events:
                if new_trace + Trace([a]) not in self.obs_table.get_low_S() \
                        and new_trace + Trace([a]) not in self.obs_table.get_S():
                    self.obs_table.get_low_S().append(new_trace + Trace([a]))
                    new_state: List[State] = []
                    # add empty cells to T
                    for j in range(len(self.obs_table.get_E())):
                        new_state.append(State([(None, None)]))
                    low_obs.append(Row(new_state))

        self.obs_table.set_upper_observations(upp_obs)
        self.obs_table.set_lower_observations(low_obs)

    @staticmethod
    def get_nondetermistic_edge(sha: StochasticHybridAutomaton, loc: Location):
        outgoing_edges = [e for e in sha.edges if e.start == loc]
        seen_events: Set[str] = set()
        for edge in outgoing_edges:
            if edge.sync in seen_events:
                LOGGER.warn('NON-DETERMINISM DETECTED! Location: {}, Event: {}'.format(loc.name, edge.sync))
                return edge.sync
            else:
                seen_events.add(edge.sync)
        return None

    def merge_loc(self, sha: StochasticHybridAutomaton, loc: Location,
                  event: str, loc_dic: Dict[Trace, str]):
        competing_locs = [edge.dest for edge in sha.edges if edge.start == loc and edge.sync == event]
        # If ambiguous rows have different flow/distr, they cannot be merged.
        if len(set([l.flow_cond for l in competing_locs])) > 1:
            return sha, False

        ambig_traces = [tr for tr in loc_dic.keys() for l in competing_locs if loc_dic[tr] == l.name]
        ambig_rows = [row for i, row in enumerate(self.obs_table.get_upper_observations()) for tr in ambig_traces if
                      self.obs_table.get_S().index(tr) == i]

        # if two among the competing rows are not at least weakly equal,
        # merging is not possible
        checked_pairs: List[Tuple[Row, Row]] = []
        for i, row_1 in enumerate(ambig_rows):
            for j, row_2 in enumerate(ambig_rows):
                if i != j and (row_1, row_2) not in checked_pairs and (row_2, row_1) not in checked_pairs:
                    if not self.TEACHER.eqr_query(row_1, row_2):
                        return sha, False
                    else:
                        checked_pairs.append((row_1, row_2))

        # otherwise, merge.
        to_remove = [e for e in sha.edges if e.dest in competing_locs[1:] and e.start == loc]
        starting_in_competing = [e for e in sha.edges if e.start in competing_locs[1:]]
        ending_in_competing = [e for e in sha.edges if e.dest in competing_locs[1:] and e.start != loc]
        for e in to_remove:
            sha.edges.remove(e)
        for e in starting_in_competing:
            sha.edges.append(Edge(competing_locs[0], e.dest, e.guard, e.sync))
            sha.edges.remove(e)
        for e in ending_in_competing:
            sha.edges.append(Edge(e.start, competing_locs[0], e.guard, e.sync))
            try:
                sha.edges.remove(e)
            except ValueError:
                LOGGER.warn('Edge already removed.')

        for l in competing_locs[1:]:
            try:
                sha.locations.remove(l)
            except ValueError:
                LOGGER.warn('Location already removed.')

        return sha, True

    def sanity_check(self, sha: StochasticHybridAutomaton, loc_dic: Dict[Trace, str]):
        to_check: Set[Location] = set(sha.locations)

        while len(to_check) > 0:
            for loc in sha.locations:
                non_det_event = Learner.get_nondetermistic_edge(sha, loc)
                if non_det_event is not None:
                    sha, merged = self.merge_loc(sha, loc, non_det_event, loc_dic)
                    if merged:
                        to_check = set(sha.locations)
                        break
                    else:
                        LOGGER.warn('MERGING LOCATIONS UNSUCCESSFUL.')
                        to_check = set()
                        break
                else:
                    to_check.remove(loc)

        return sha

    def run_lsha(self, debug_print=True, filter_empty=False):
        # Fill Observation Table with Answers to Queries (from TEACHER)
        step0 = True  # work around to implement a do-while structure

        self.fill_table()

        self.TEACHER.ref_query(self.obs_table)
        self.fill_table()

        counterexample = self.TEACHER.get_counterexample(self.obs_table)

        while counterexample is not None or step0:
            step0 = False
            # if a counterexample was found, it (and all of its prefixes) are added to set S
            if counterexample is not None:
                LOGGER.warn('FOUND COUNTEREXAMPLE: {}'.format(counterexample))
                self.add_counterexample(counterexample)
                self.fill_table()

            if debug_print:
                LOGGER.info('OBSERVATION TABLE')
                self.obs_table.print(filter_empty)

            # Check if obs. table is closed
            closedness_check = self.is_closed()
            consistency_check, discriminating_symbol = self.is_consistent()
            while not (closedness_check and consistency_check):
                if not closedness_check:
                    LOGGER.warn('!!TABLE IS NOT CLOSED!!')
                    # If not, make closed
                    self.make_closed()
                    self.fill_table()
                    LOGGER.info('CLOSED OBSERVATION TABLE')
                    self.obs_table.print(filter_empty)

                # Check if obs. table is consistent
                if not consistency_check:
                    LOGGER.warn('!!TABLE IS NOT CONSISTENT!!')
                    # If not, make consistent
                    self.make_consistent(discriminating_symbol)
                    self.fill_table()
                    LOGGER.info('CONSISTENT OBSERVATION TABLE')
                    self.obs_table.print(filter_empty)

                closedness_check = self.is_closed()
                consistency_check, discriminating_symbol = self.is_consistent()

            self.TEACHER.ref_query(self.obs_table)
            self.fill_table()
            if debug_print:
                LOGGER.info('OBSERVATION TABLE')
                self.obs_table.print(filter_empty)
            counterexample = self.TEACHER.get_counterexample(self.obs_table)

        if debug_print:
            LOGGER.info('FINAL OBSERVATION TABLE')
            self.obs_table.print(filter_empty)
        # Build Hypothesis Automaton
        LOGGER.info('BUILDING HYP. AUTOMATON...')
        hypsha, loc_dict = self.obs_table.to_sha(self.TEACHER)
        try:
            hypsha = self.sanity_check(hypsha, loc_dict)
        except:
            LOGGER.error("Error occurred while fixing the SHA.")
        return hypsha
