from typing import List, Tuple

from it.polimi.hri_learn.domain.hafeatures import HybridAutomaton, Location, Edge
from it.polimi.hri_learn.lstar_sha.logger import Logger
import configparser
import sys

config = configparser.ConfigParser()
config.sections()
config.read(sys.argv[1])
config.sections()

EMPTY_STRING = '\u03B5'

MODEL_FORMATTER = 'f_{}'
DISTR_FORMATTER = 'N_{}'
LOCATION_FORMATTER = 'q_{}'
LOGGER = Logger()


class ObsTable:
    def __init__(self, s: List[str], e: List[str], low_s: List[str]):
        self.__S = s
        self.__low_S = low_s
        self.__E = e
        self.__upp_obs: List[List[Tuple]] = [[(None, None)] * len(e)] * len(s)
        self.__low_obs: List[List[Tuple]] = [[(None, None)] * len(e)] * len(low_s)

    def get_S(self):
        return self.__S

    def add_S(self, word: str):
        self.__S.append(word)

    def get_E(self):
        return self.__E

    def add_E(self, word: str):
        self.__E.append(word)

    def get_low_S(self):
        return self.__low_S

    def add_low_S(self, word: str):
        self.__low_S.append(word)

    def del_low_S(self, index: int):
        self.get_low_S().pop(index)

    def get_upper_observations(self):
        return self.__upp_obs

    def set_upper_observations(self, obs_table: List[List[Tuple]]):
        self.__upp_obs = obs_table

    def get_lower_observations(self):
        return self.__low_obs

    def set_lower_observations(self, obs_table: List[List[Tuple]]):
        self.__low_obs = obs_table

    @staticmethod
    def tuple_to_str(tup):
        if tup[0] is None and tup[1] is None:
            return '(∅, ∅)\t'
        else:
            return '({}, {})'.format(MODEL_FORMATTER.format(tup[0]), DISTR_FORMATTER.format(tup[1]))

    def to_str(self, filter_empty=False):
        result = ''
        max_s = max([len(word) / 3 for word in self.get_S()])
        max_low_s = max([len(word) / 3 for word in self.get_low_S()])
        max_tabs = int(max(max_s, max_low_s))

        HEADER = '\t' * max_tabs + '|\t\t'
        for t_word in self.get_E():
            HEADER += t_word if t_word != '' else EMPTY_STRING
            HEADER += '\t\t|\t\t'
        result += HEADER + '\n'

        SEPARATOR = '----' * max_tabs + '+' + '---------------+' * len(self.get_E())
        result += SEPARATOR + '\n'

        for (i, s_word) in enumerate(self.get_S()):
            row = self.get_upper_observations()[i]
            row_is_populated = any([row[j][0] is not None and row[j][1] is not None for j in range(len(self.get_E()))])
            if filter_empty and not row_is_populated:
                pass
            else:
                ROW = s_word if s_word != '' else EMPTY_STRING
                len_word = int(len(s_word) / 3) if s_word != '' else 1
                ROW += '\t' * (max_tabs + 1 - len_word) + '|\t' if len_word < max_tabs - 1 or max_tabs <= 4 \
                    else '\t' * (max_tabs + 2 - len_word) + '|\t'
                for (j, t_word) in enumerate(self.get_E()):
                    ROW += ObsTable.tuple_to_str(self.get_upper_observations()[i][j])
                    ROW += '\t|\t'
                result += ROW + '\n'
        result += SEPARATOR + '\n'
        for (i, s_word) in enumerate(self.get_low_S()):
            row = self.get_lower_observations()[i]
            row_is_populated = any([row[j][0] is not None and row[j][1] is not None for j in range(len(self.get_E()))])
            if filter_empty and not row_is_populated:
                pass
            else:
                ROW = s_word if s_word != '' else EMPTY_STRING
                len_word = int(len(s_word) / 3)
                ROW += '\t' * (max_tabs + 1 - len_word) + '|\t' if len_word < max_tabs - 1 or max_tabs <= 4 \
                    else '\t' * (max_tabs + 2 - len_word) + '|\t'
                for (j, t_word) in enumerate(self.get_E()):
                    ROW += ObsTable.tuple_to_str(self.get_lower_observations()[i][j])
                    ROW += '\t|\t'
                result += ROW + '\n'
        result += SEPARATOR + '\n'
        return result

    def print(self, filter_empty=False):
        print(self.to_str(filter_empty))


class Learner:
    def __init__(self, teacher, table: ObsTable = None):
        self.symbols = teacher.get_symbols()
        self.TEACHER = teacher
        self.obs_table = table if table is not None else ObsTable([''], [''], list(self.symbols.keys()))

    def set_symbols(self, symbols):
        self.symbols = symbols

    def get_symbols(self):
        return self.symbols

    def set_table(self, table: ObsTable):
        self.obs_table = table

    def get_table(self):
        return self.obs_table

    def fill_table(self):
        upp_obs: List[List[Tuple]] = self.get_table().get_upper_observations()
        for (i, s_word) in enumerate(self.get_table().get_S()):
            row: List[Tuple] = upp_obs[i].copy()
            for (j, t_word) in enumerate(self.get_table().get_E()):
                # if cell is yet to be filled,
                # asks teacher to answer queries
                # and fills cell with answers
                if upp_obs[i][j][0] is None:
                    identified_model = self.TEACHER.mi_query(s_word + t_word)
                    identified_distr = self.TEACHER.ht_query(s_word + t_word, identified_model)
                    if identified_model is None or identified_distr is None:
                        identified_model = None
                        identified_distr = None
                    cell = (identified_model, identified_distr)
                    row[j] = cell
                    if cell == (None, None) and j == 0:
                        break
            upp_obs[i] = row.copy()
        self.get_table().set_upper_observations(upp_obs)

        low_obs: List[List[Tuple]] = self.get_table().get_lower_observations()
        for (i, s_word) in enumerate(self.get_table().get_low_S()):
            row: List[Tuple] = low_obs[i].copy()
            for (j, t_word) in enumerate(self.get_table().get_E()):
                # if cell is yet to be filled,
                # asks teacher to answer queries
                # and fills cell with answers
                go_on = j == 0 or (j > 0 and low_obs[i][0] != (None, None))
                if go_on and low_obs[i][j] == (None, None):
                    identified_model = self.TEACHER.mi_query(s_word + t_word)
                    identified_distr = self.TEACHER.ht_query(s_word + t_word, identified_model)
                    if identified_model is None or identified_distr is None:
                        identified_model = None
                        identified_distr = None
                    cell = (identified_model, identified_distr)
                    row[j] = cell
            low_obs[i] = row.copy()
        self.get_table().set_lower_observations(low_obs)

    def is_closed(self):
        upp_obs = self.get_table().get_upper_observations()
        low_obs = self.get_table().get_lower_observations()
        for (l_i, row) in enumerate(low_obs):
            row_is_filled = row[0] != (None, None)
            if not row_is_filled:
                continue
            row_is_in_upper = False
            for (s_i, s_word) in enumerate(self.get_table().get_S()):
                if self.TEACHER.eqr_query(self.get_table().get_low_S()[l_i], s_word, row, upp_obs[s_i]):
                    row_is_in_upper = True
                    break
            if not row_is_in_upper:
                return False
        else:
            return True

    def is_consistent(self, symbols):
        upp_obs = self.get_table().get_upper_observations()
        pairs: List[Tuple] = []
        # FIXME: each pair shows up twice, duplicates should be cleared
        for (index, row) in enumerate(upp_obs):
            equal_rows = [i for (i, r) in enumerate(upp_obs) if index != i and r == row]
            S = self.get_table().get_S()
            equal_pairs = [(S[index], S[equal_i]) for equal_i in equal_rows]
            pairs += equal_pairs
        if len(pairs) == 0:
            return True, None
        else:
            for pair in pairs:
                for symbol in symbols.keys():
                    try:
                        new_pair_1 = self.get_table().get_S().index(pair[0] + symbol)
                        new_row_1 = self.get_table().get_upper_observations()[new_pair_1]
                    except ValueError:
                        new_pair_1 = self.get_table().get_low_S().index(pair[0] + symbol)
                        new_row_1 = self.get_table().get_lower_observations()[new_pair_1]

                    try:
                        new_pair_2 = self.get_table().get_S().index(pair[1] + symbol)
                        new_row_2 = self.get_table().get_upper_observations()[new_pair_2]
                    except ValueError:
                        new_pair_2 = self.get_table().get_low_S().index(pair[1] + symbol)
                        new_row_2 = self.get_table().get_lower_observations()[new_pair_2]

                    new_1_populated = all([new_row_1[i][0] is not None and new_row_1[i][1] is not None
                                           for i in range(len(self.get_table().get_E()))])
                    new_2_populated = all([new_row_2[i][0] is not None and new_row_2[i][1] is not None
                                           for i in range(len(self.get_table().get_E()))])

                    rows_different = not self.TEACHER.eqr_query(pair[0] + symbol, pair[1] + symbol, new_row_1,
                                                                new_row_2)
                    if new_1_populated and new_2_populated and rows_different:
                        for (e_i, e_word) in enumerate(self.get_table().get_E()):
                            if new_row_1[e_i] != new_row_2[e_i]:
                                LOGGER.warn('INCONSISTENCY: {}-{}'.format(pair[0] + symbol, pair[1] + symbol))
                                return False, symbol + e_word
            else:
                return True, None

    def make_closed(self):
        upp_obs: List[List[Tuple]] = self.get_table().get_upper_observations()
        low_S = self.get_table().get_low_S()
        low_obs: List[List[Tuple]] = self.get_table().get_lower_observations()
        for (index, row) in enumerate(low_obs):
            row_is_populated = any([cell[0] is not None and cell[1] is not None for cell in row])
            # if there is a populated row in lower portion that is not in the upper portion
            # the corresponding word is added to the S word set
            row_present = False
            for (s_i, s_word) in enumerate(self.get_table().get_S()):
                if self.TEACHER.eqr_query(low_S[index], self.get_table().get_S()[s_i], row, upp_obs[s_i]):
                    row_present = True
                    break
            if row_is_populated and not row_present:
                upp_obs.append(row)
                new_s_word = low_S[index]
                self.get_table().add_S(new_s_word)
                low_obs.pop(index)
                self.get_table().del_low_S(index)
                # lower portion is then updated with all combinations of
                # new S word and all possible symbols
                for symbol in self.get_symbols():
                    self.get_table().add_low_S(new_s_word + symbol)
                    new_row: List[Tuple] = [(None, None)] * len(self.get_table().get_E())
                    low_obs.append(new_row)
        self.get_table().set_upper_observations(upp_obs)
        self.get_table().set_lower_observations(low_obs)
        self.fill_table()

    def make_consistent(self, discr_sym: str):
        self.get_table().add_E(discr_sym)
        upp_obs = self.get_table().get_upper_observations()
        low_obs = self.get_table().get_lower_observations()
        for s_i in range(len(upp_obs)):
            upp_obs[s_i].append((None, None))
        for s_i in range(len(low_obs)):
            low_obs[s_i].append((None, None))
        self.fill_table()

    def add_counterexample(self, counterexample: str):
        upp_obs = self.get_table().get_upper_observations()
        low_obs = self.get_table().get_lower_observations()

        # add counterexample and all its prefixes to S
        for i in range(3, len(counterexample) + 1, 3):
            if counterexample[:i] not in self.get_table().get_S():
                self.get_table().get_S().append(counterexample[:i])
                upp_obs.append([])
                # add empty cells to T
                for j in range(len(self.get_table().get_E())):
                    upp_obs[len(self.get_table().get_S()) - 1].append((None, None))

            if counterexample[:i] in self.get_table().get_low_S():
                row_index = self.get_table().get_low_S().index(counterexample[:i])
                self.get_table().get_lower_observations().pop(row_index)
                self.get_table().get_low_S().pop(row_index)

            # add 1-step away words to low_S
            for a in self.get_symbols():
                if counterexample[:i] + a not in self.get_table().get_low_S() \
                        and counterexample[:i] + a not in self.get_table().get_S():
                    self.get_table().get_low_S().append(counterexample[:i] + a)
                    low_obs.append([])
                    # add empty cells to T
                    for j in range(len(self.get_table().get_E())):
                        low_obs[len(self.get_table().get_low_S()) - 1].append((None, None))

    def build_hyp_aut(self):
        locations: List[Location] = []
        upp_obs = self.get_table().get_upper_observations()
        low_obs: List[List[Tuple]] = self.get_table().get_lower_observations()
        unique_sequences: List[List[Tuple]] = []
        for (i, row) in enumerate(upp_obs):
            row_already_present = False
            for seq in unique_sequences:
                s_word = self.get_table().get_S()[upp_obs.index(seq)]
                if self.TEACHER.eqr_query(s_word, self.get_table().get_S()[i], seq, row):
                    row_already_present = True
                    break
            if not row_already_present:
                unique_sequences.append(row)
        for (index, seq) in enumerate(unique_sequences):
            new_name = LOCATION_FORMATTER.format(index)
            new_flow = MODEL_FORMATTER.format(seq[0][0]) + ', ' + DISTR_FORMATTER.format(seq[0][1])
            locations.append(Location(new_name, new_flow))

        edges: List[Edge] = []
        for (s_i, s_word) in enumerate(self.get_table().get_S()):
            for (t_i, t_word) in enumerate(self.get_table().get_E()):
                if upp_obs[s_i][t_i][0] is not None and upp_obs[s_i][t_i][1] is not None:
                    word: str = s_word + t_word
                    entry_word = word[:-3] if t_word != '' else s_word[:-3]
                    try:
                        start_row_index = self.get_table().get_S().index(entry_word)
                        start_row = unique_sequences.index(upp_obs[start_row_index])
                    except ValueError:
                        if entry_word in self.get_table().get_low_S():
                            start_row_index = self.get_table().get_low_S().index(entry_word)
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
                            s2 = self.get_table().get_S()[upp_obs.index(seq)]
                            if self.TEACHER.eqr_query(s_word, s2, upp_obs[s_i], seq):
                                eq_rows.append(seq)
                        eq_row = eq_rows[0]
                        dest_row = unique_sequences.index(eq_row)
                        dest_loc = locations[dest_row]
                    else:
                        try:
                            dest_row_index = self.get_table().get_S().index(word)
                            eq_rows = []
                            for seq in unique_sequences:
                                s1 = self.get_table().get_S()[dest_row_index]
                                s2 = self.get_table().get_S()[upp_obs.index(seq)]
                                if self.TEACHER.eqr_query(s1, s2, upp_obs[dest_row_index], seq):
                                    eq_rows.append(seq)
                            eq_row = eq_rows[0]
                        except ValueError:
                            if word in self.get_table().get_low_S():
                                dest_row_index = self.get_table().get_low_S().index(word)
                                eq_rows = []
                                for seq in unique_sequences:
                                    s1 = self.get_table().get_low_S()[dest_row_index]
                                    s2 = self.get_table().get_S()[upp_obs.index(seq)]
                                    if self.TEACHER.eqr_query(s1, s2, low_obs[dest_row_index], seq):
                                        eq_rows.append(seq)
                                eq_row = eq_rows[0]
                            else:
                                continue
                        dest_row = unique_sequences.index(eq_row)
                        dest_loc = locations[dest_row]
                    # labels = self.get_symbols()[word[-3:]].split(' and ') if word != '' else ['', EMPTY_STRING]
                    labels = word[-3:] if word != '' else EMPTY_STRING
                    new_edge = Edge(start_loc, dest_loc, sync=labels)  # guard=labels[0], sync=labels[1])
                    if new_edge not in edges:
                        edges.append(new_edge)

        for (s_i, s_word) in enumerate(self.get_table().get_low_S()):
            for (t_i, t_word) in enumerate(self.get_table().get_E()):
                if low_obs[s_i][t_i][0] is not None and low_obs[s_i][t_i][1] is not None:
                    word = s_word + t_word
                    entry_word = word[:-3]
                    try:
                        start_row_index = self.get_table().get_S().index(entry_word)
                        start_row = unique_sequences.index(upp_obs[start_row_index])
                    except ValueError:
                        if entry_word in self.get_table().get_low_S():
                            start_row_index = self.get_table().get_low_S().index(entry_word)
                            eq_rows = []
                            for seq in unique_sequences:
                                s1 = self.get_table().get_S()[upp_obs.index(seq)]
                                s2 = self.get_table().get_low_S()[start_row_index]
                                if self.TEACHER.eqr_query(s1, s2, seq, low_obs[start_row_index]):
                                    eq_rows.append(seq)
                            start_row = unique_sequences.index(eq_rows[0])
                        else:
                            continue
                    start_loc = locations[start_row]
                    try:
                        dest_row_index = self.get_table().get_S().index(word)
                        eq_rows = []
                        for seq in unique_sequences:
                            s1 = self.get_table().get_S()[dest_row_index]
                            s2 = self.get_table().get_S()[upp_obs.index(seq)]
                            if self.TEACHER.eqr_query(s1, s2, upp_obs[dest_row_index], seq):
                                eq_rows.append(seq)
                        eq_row = eq_rows[0]
                    except ValueError:
                        if word in self.get_table().get_low_S():
                            dest_row_index = self.get_table().get_low_S().index(word)
                            eq_rows = []
                            for seq in unique_sequences:
                                s1 = self.get_table().get_low_S()[dest_row_index]
                                s2 = self.get_table().get_S()[upp_obs.index(seq)]
                                if self.TEACHER.eqr_query(s1, s2, low_obs[dest_row_index], seq):
                                    eq_rows.append(seq)
                            eq_row = eq_rows[0]
                        else:
                            continue
                    dest_loc = locations[unique_sequences.index(eq_row)]
                    if word != '':
                        # labels = self.get_symbols()[word.replace(entry_word, '')].split(' and ')
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
        self.TEACHER.ref_query(self.get_table())
        self.fill_table()
        counterexample = self.TEACHER.get_counterexample(self.get_table())
        while counterexample is not None or step0:
            if config['DEFAULT']['PLOT_DISTR'] == 'True':
                self.TEACHER.plot_distributions()
            step0 = False
            if counterexample is not None:
                LOGGER.warn('FOUND COUNTEREXAMPLE: {}'.format(counterexample))
                self.add_counterexample(counterexample)
                self.fill_table()
            self.TEACHER.ref_query(self.get_table())
            self.fill_table()

            if debug_print:
                LOGGER.info('OBSERVATION TABLE')
                self.get_table().print(filter_empty)

            # Check if obs. table is closed
            closedness_check = self.is_closed()
            consistency_check, discriminating_symbol = self.is_consistent(self.get_symbols())
            while not (closedness_check and consistency_check):
                if not closedness_check:
                    LOGGER.warn('!!TABLE IS NOT CLOSED!!')
                    # If not, make closed
                    self.make_closed()
                    LOGGER.msg('CLOSED OBSERVATION TABLE')
                    self.get_table().print(filter_empty)

                # Check if obs. table is consistent
                if not consistency_check:
                    LOGGER.warn('!!TABLE IS NOT CONSISTENT!!')
                    # If not, make consistent
                    self.make_consistent(discriminating_symbol)
                    LOGGER.msg('CONSISTENT OBSERVATION TABLE')
                    self.get_table().print(filter_empty)

                closedness_check = self.is_closed()
                consistency_check, discriminating_symbol = self.is_consistent(self.get_symbols())

            counterexample = self.TEACHER.get_counterexample(self.get_table())

        if debug_print:
            LOGGER.msg('FINAL OBSERVATION TABLE')
            self.get_table().print(filter_empty)
        # Build Hypothesis Automaton
        LOGGER.info('BUILDING HYP. AUTOMATON...')
        return self.build_hyp_aut()
