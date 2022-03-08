import configparser
from typing import List, Tuple

from it.polimi.hri_learn.domain.lshafeatures import State, FlowCondition, ProbDistribution
from it.polimi.hri_learn.domain.obstable import ObsTable, Row, Trace
from it.polimi.hri_learn.lstar_sha.logger import Logger
from it.polimi.hri_learn.lstar_sha.teacher import Teacher

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
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
                            if new_row_1.state[e_i] != new_row_2.state[e_i]:
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

    def run_lsha(self, debug_print=True, filter_empty=False):
        # Fill Observation Table with Answers to Queries (from TEACHER)
        step0 = True  # work around to implement a do-while structure

        self.fill_table()

        self.TEACHER.ref_query(self.obs_table)
        self.fill_table()

        counterexample = self.TEACHER.get_counterexample(self.obs_table)

        while counterexample is not None or step0:
            step0 = False
            # plots currently known distributions
            if config['DEFAULT']['PLOT_DISTR'] == 'True':
                self.TEACHER.sul.plot_distributions()
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
                    LOGGER.msg('CLOSED OBSERVATION TABLE')
                    self.obs_table.print(filter_empty)

                # Check if obs. table is consistent
                if not consistency_check:
                    LOGGER.warn('!!TABLE IS NOT CONSISTENT!!')
                    # If not, make consistent
                    self.make_consistent(discriminating_symbol)
                    self.fill_table()
                    LOGGER.msg('CONSISTENT OBSERVATION TABLE')
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
            LOGGER.msg('FINAL OBSERVATION TABLE')
            self.obs_table.print(filter_empty)
        # Build Hypothesis Automaton
        LOGGER.info('BUILDING HYP. AUTOMATON...')
        return self.obs_table.to_sha(self.TEACHER)
