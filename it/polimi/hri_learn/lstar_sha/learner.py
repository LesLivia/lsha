import configparser
from typing import List, Tuple

from it.polimi.hri_learn.domain.hafeatures import HybridAutomaton, Location, Edge
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

    def fill_row(self, row: Row, i: int, s_word: str, obs: List[Row]):
        for (j, t_word) in enumerate(self.obs_table.get_E()):
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
        for (i, s_word) in enumerate(self.obs_table.get_S()):
            row: Row = Row(upp_obs[i].state.copy())
            row = self.fill_row(row, i, s_word, upp_obs)
            upp_obs[i] = row
        self.obs_table.set_upper_observations(upp_obs)

        low_obs: List[Row] = self.obs_table.get_lower_observations()
        for (i, s_word) in enumerate(self.obs_table.get_low_S()):
            row: Row = Row(low_obs[i].state.copy())
            row = self.fill_row(row, i, s_word, low_obs)
            low_obs[i] = row
        self.obs_table.set_lower_observations(low_obs)

    def is_closed(self):
        upp_obs = self.obs_table.get_upper_observations()
        low_obs = self.obs_table.get_lower_observations()
        for (l_i, row) in enumerate(low_obs):
            if not row.is_populated():
                continue
            row_is_in_upper = False
            for (s_i, s_word) in enumerate(self.obs_table.get_S()):
                if self.TEACHER.eqr_query(row, upp_obs[s_i]):
                    row_is_in_upper = True
                    break
            if not row_is_in_upper:
                return False
        else:
            return True

    def is_consistent(self, symbols):
        upp_obs = self.obs_table.get_upper_observations()
        pairs: List[Tuple] = []
        # FIXME: each pair shows up twice, duplicates should be cleared
        for (index, row) in enumerate(upp_obs):
            equal_rows = [i for (i, r) in enumerate(upp_obs) if index != i and r == row]
            S = self.obs_table.get_S()
            equal_pairs = [(S[index], S[equal_i]) for equal_i in equal_rows]
            pairs += equal_pairs
        if len(pairs) == 0:
            return True, None
        else:
            for pair in pairs:
                for symbol in symbols.keys():
                    try:
                        new_pair_1 = self.obs_table.get_S().index(pair[0] + symbol)
                        new_row_1 = self.obs_table.get_upper_observations()[new_pair_1]
                    except ValueError:
                        new_pair_1 = self.obs_table.get_low_S().index(pair[0] + symbol)
                        new_row_1 = self.obs_table.get_lower_observations()[new_pair_1]

                    try:
                        new_pair_2 = self.obs_table.get_S().index(pair[1] + symbol)
                        new_row_2 = self.obs_table.get_upper_observations()[new_pair_2]
                    except ValueError:
                        new_pair_2 = self.obs_table.get_low_S().index(pair[1] + symbol)
                        new_row_2 = self.obs_table.get_lower_observations()[new_pair_2]

                    new_1_populated = all([new_row_1[i][0] is not None and new_row_1[i][1] is not None
                                           for i in range(len(self.obs_table.get_E()))])
                    new_2_populated = all([new_row_2[i][0] is not None and new_row_2[i][1] is not None
                                           for i in range(len(self.obs_table.get_E()))])

                    rows_different = not self.TEACHER.eqr_query(new_row_1, new_row_2)
                    if new_1_populated and new_2_populated and rows_different:
                        for (e_i, e_word) in enumerate(self.obs_table.get_E()):
                            if new_row_1[e_i] != new_row_2[e_i]:
                                LOGGER.warn('INCONSISTENCY: {}-{}'.format(pair[0] + symbol, pair[1] + symbol))
                                return False, symbol + e_word
            else:
                return True, None

    def make_closed(self):
        upp_obs: List[Row] = self.obs_table.get_upper_observations()
        low_S = self.obs_table.get_low_S()
        low_obs: List[Row] = self.obs_table.get_lower_observations()
        for index, row in enumerate(low_obs):
            # if there is a populated row in lower portion that is not in the upper portion
            # the corresponding word is added to the S word set
            row_present = any([self.TEACHER.eqr_query(row, row_2) for row_2 in upp_obs])
            if row.is_populated() and not row_present:
                upp_obs.append(row)
                new_s_word = low_S[index]
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
        upp_obs = self.obs_table.get_upper_observations()
        low_obs = self.obs_table.get_lower_observations()
        for s_i in range(len(upp_obs)):
            upp_obs[s_i].append((None, None))
        for s_i in range(len(low_obs)):
            low_obs[s_i].append((None, None))

    def add_counterexample(self, counterexample: str):
        upp_obs = self.obs_table.get_upper_observations()
        low_obs = self.obs_table.get_lower_observations()

        # add counterexample and all its prefixes to S
        for i in range(3, len(counterexample) + 1, 3):
            if counterexample[:i] not in self.obs_table.get_S():
                self.obs_table.get_S().append(counterexample[:i])
                upp_obs.append([])
                # add empty cells to T
                for j in range(len(self.obs_table.get_E())):
                    upp_obs[len(self.obs_table.get_S()) - 1].append((None, None))

            if counterexample[:i] in self.obs_table.get_low_S():
                row_index = self.obs_table.get_low_S().index(counterexample[:i])
                self.obs_table.get_lower_observations().pop(row_index)
                self.obs_table.get_low_S().pop(row_index)

            # add 1-step away words to low_S
            for a in self.symbols:
                if counterexample[:i] + a not in self.obs_table.get_low_S() \
                        and counterexample[:i] + a not in self.obs_table.get_S():
                    self.obs_table.get_low_S().append(counterexample[:i] + a)
                    low_obs.append([])
                    # add empty cells to T
                    for j in range(len(self.obs_table.get_E())):
                        low_obs[len(self.obs_table.get_low_S()) - 1].append((None, None))

    def build_hyp_aut(self):
        locations: List[Location] = []
        upp_obs = self.obs_table.get_upper_observations()
        low_obs: List[List[Tuple]] = self.obs_table.get_lower_observations()
        unique_sequences: List[List[Tuple]] = []
        for (i, row) in enumerate(upp_obs):
            row_already_present = False
            for seq in unique_sequences:
                s_word = self.obs_table.get_S()[upp_obs.index(seq)]
                if self.TEACHER.eqr_query(seq, row):
                    row_already_present = True
                    break
            if not row_already_present:
                unique_sequences.append(row)
        for (index, seq) in enumerate(unique_sequences):
            new_name = LOCATION_FORMATTER.format(index)
            new_flow = MODEL_FORMATTER.format(seq[0][0]) + ', ' + DISTR_FORMATTER.format(seq[0][1])
            locations.append(Location(new_name, new_flow))

        edges: List[Edge] = []
        for (s_i, s_word) in enumerate(self.obs_table.get_S()):
            for (t_i, t_word) in enumerate(self.obs_table.get_E()):
                if upp_obs[s_i][t_i][0] is not None and upp_obs[s_i][t_i][1] is not None:
                    word: str = s_word + t_word
                    entry_word = word[:-3] if t_word != '' else s_word[:-3]
                    try:
                        start_row_index = self.obs_table.get_S().index(entry_word)
                        start_row = unique_sequences.index(upp_obs[start_row_index])
                    except ValueError:
                        if entry_word in self.obs_table.get_low_S():
                            start_row_index = self.obs_table.get_low_S().index(entry_word)
                            row_is_filled = all([cell != (None, None) for cell in low_obs[start_row_index]])
                            if row_is_filled:
                                start_row = unique_sequences.index(low_obs[start_row_index])
                            else:
                                continue
                        else:
                            continue
                    start_loc = locations[start_row]
                    if t_word == '':
                        eq_rows = []
                        for seq in unique_sequences:
                            s2 = self.obs_table.get_S()[upp_obs.index(seq)]
                            if self.TEACHER.eqr_query(s_word, s2, upp_obs[s_i], seq):
                                eq_rows.append(seq)
                        eq_row = eq_rows[0]
                        dest_row = unique_sequences.index(eq_row)
                        dest_loc = locations[dest_row]
                    else:
                        try:
                            dest_row_index = self.obs_table.get_S().index(word)
                            eq_rows = []
                            for seq in unique_sequences:
                                s1 = self.obs_table.get_S()[dest_row_index]
                                s2 = self.obs_table.get_S()[upp_obs.index(seq)]
                                if self.TEACHER.eqr_query(s1, s2, upp_obs[dest_row_index], seq):
                                    eq_rows.append(seq)
                            eq_row = eq_rows[0]
                        except ValueError:
                            if word in self.obs_table.get_low_S():
                                dest_row_index = self.obs_table.get_low_S().index(word)
                                eq_rows = []
                                for seq in unique_sequences:
                                    s1 = self.obs_table.get_low_S()[dest_row_index]
                                    s2 = self.obs_table.get_S()[upp_obs.index(seq)]
                                    if self.TEACHER.eqr_query(s1, s2, low_obs[dest_row_index], seq):
                                        eq_rows.append(seq)
                                eq_row = eq_rows[0]
                            else:
                                continue
                        dest_row = unique_sequences.index(eq_row)
                        dest_loc = locations[dest_row]
                    # labels = self.symbols[word[-3:]].split(' and ') if word != '' else ['', EMPTY_STRING]
                    labels = word[-3:] if word != '' else EMPTY_STRING
                    new_edge = Edge(start_loc, dest_loc, sync=labels)  # guard=labels[0], sync=labels[1])
                    if new_edge not in edges:
                        edges.append(new_edge)

        for (s_i, s_word) in enumerate(self.obs_table.get_low_S()):
            for (t_i, t_word) in enumerate(self.obs_table.get_E()):
                if low_obs[s_i][t_i][0] is not None and low_obs[s_i][t_i][1] is not None:
                    word = s_word + t_word
                    entry_word = word[:-3]
                    try:
                        start_row_index = self.obs_table.get_S().index(entry_word)
                        start_row = unique_sequences.index(upp_obs[start_row_index])
                    except ValueError:
                        if entry_word in self.obs_table.get_low_S():
                            start_row_index = self.obs_table.get_low_S().index(entry_word)
                            eq_rows = []
                            for seq in unique_sequences:
                                s1 = self.obs_table.get_S()[upp_obs.index(seq)]
                                s2 = self.obs_table.get_low_S()[start_row_index]
                                if self.TEACHER.eqr_query(s1, s2, seq, low_obs[start_row_index]):
                                    eq_rows.append(seq)
                            start_row = unique_sequences.index(eq_rows[0])
                        else:
                            continue
                    start_loc = locations[start_row]
                    try:
                        dest_row_index = self.obs_table.get_S().index(word)
                        eq_rows = []
                        for seq in unique_sequences:
                            s1 = self.obs_table.get_S()[dest_row_index]
                            s2 = self.obs_table.get_S()[upp_obs.index(seq)]
                            if self.TEACHER.eqr_query(s1, s2, upp_obs[dest_row_index], seq):
                                eq_rows.append(seq)
                        eq_row = eq_rows[0]
                    except ValueError:
                        if word in self.obs_table.get_low_S():
                            dest_row_index = self.obs_table.get_low_S().index(word)
                            eq_rows = []
                            for seq in unique_sequences:
                                s1 = self.obs_table.get_low_S()[dest_row_index]
                                s2 = self.obs_table.get_S()[upp_obs.index(seq)]
                                if self.TEACHER.eqr_query(s1, s2, low_obs[dest_row_index], seq):
                                    eq_rows.append(seq)
                            eq_row = eq_rows[0]
                        else:
                            continue
                    dest_loc = locations[unique_sequences.index(eq_row)]
                    if word != '':
                        # labels = self.symbols[word.replace(entry_word, '')].split(' and ')
                        labels = word.replace(entry_word, '')
                    else:
                        # labels = ['', EMPTY_STRING]
                        labels = EMPTY_STRING
                    new_edge = Edge(start_loc, dest_loc, sync=labels)  # guard=labels[0], sync=labels[1])
                    if new_edge not in edges:
                        edges.append(new_edge)

        return HybridAutomaton(locations, edges)

    def run_lsha(self, debug_print=True, filter_empty=False):
        # Fill Observation Table with Answers to Queries (from TEACHER)
        step0 = True
        self.fill_table()
        self.TEACHER.ref_query(self.obs_table)
        self.fill_table()
        counterexample = self.TEACHER.get_counterexample(self.obs_table)
        while counterexample is not None or step0:
            if config['DEFAULT']['PLOT_DISTR'] == 'True':
                self.TEACHER.sul.plot_distributions()

            if counterexample is not None:
                LOGGER.warn('FOUND COUNTEREXAMPLE: {}'.format(counterexample))
                self.add_counterexample(counterexample)
                # self.fill_table()
            if not step0:
                self.TEACHER.ref_query(self.obs_table)
                self.fill_table()

            step0 = False

            if debug_print:
                LOGGER.info('OBSERVATION TABLE')
                self.obs_table.print(filter_empty)

            # Check if obs. table is closed
            closedness_check = self.is_closed()
            consistency_check, discriminating_symbol = self.is_consistent(self.symbols)
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
                consistency_check, discriminating_symbol = self.is_consistent(self.symbols)

            counterexample = self.TEACHER.get_counterexample(self.obs_table)

        if debug_print:
            LOGGER.msg('FINAL OBSERVATION TABLE')
            self.obs_table.print(filter_empty)
        # Build Hypothesis Automaton
        LOGGER.info('BUILDING HYP. AUTOMATON...')
        return self.build_hyp_aut()
