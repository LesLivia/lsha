from typing import List, Dict

from sha_learning.domain.lshafeatures import Trace, State, EMPTY_STRING
from sha_learning.domain.shafeatures import StochasticHybridAutomaton, Location, Edge
from sha_learning.learning_setup.logger import Logger

LOGGER = Logger('Obs.Table Handler')


class Row:
    def __init__(self, state: List[State]):
        self.state = state

    def is_populated(self):
        return any([s.observed() for s in self.state])

    def __str__(self):
        return '\t|\t'.join([str(s) for s in self.state])

    def __eq__(self, other):
        return all([s == other.state[i] for i, s in enumerate(self.state)])

    def __hash__(self):
        return hash(str(self))


class ObsTable:
    def __init__(self, s: List[Trace], e: List[Trace], low_s: List[Trace]):
        self.__S: List[Trace] = s
        self.__low_S: List[Trace] = low_s
        self.__E: List[Trace] = e
        self.__upp_obs: List[Row] = [Row([State([(None, None)])] * len(e))] * len(s)
        self.__low_obs: List[Row] = [Row([State([(None, None)])] * len(e))] * len(low_s)

    def get_S(self):
        return self.__S

    def add_S(self, word: Trace):
        self.__S.append(word)

    def get_E(self):
        return self.__E

    def add_E(self, word: Trace):
        self.__E.append(word)

    def get_low_S(self):
        return self.__low_S

    def add_low_S(self, word: Trace):
        self.__low_S.append(word)

    def del_low_S(self, index: int):
        self.get_low_S().pop(index)

    def get_upper_observations(self):
        return self.__upp_obs

    def set_upper_observations(self, obs_table: List[Row]):
        self.__upp_obs = obs_table

    def get_lower_observations(self):
        return self.__low_obs

    def set_lower_observations(self, obs_table: List[Row]):
        self.__low_obs = obs_table

    def __str__(self, filter_empty=False):
        result = ''

        rows = self.get_upper_observations() + self.get_lower_observations()
        populated_rows = [i for i, row in enumerate(rows) if row.is_populated()]

        max_tabs = max(
            [len(str(word)) for i, word in enumerate(self.get_S() + self.get_low_S()) if i in populated_rows])
        HEADER = ' ' * max_tabs + '|'

        len_row_cells = [[len(s.label) for s in r.state] for r in rows]
        col_width = [max([l for r in len_row_cells for j_2, l in enumerate(r) if j_2 == j]) for j, e in
                     enumerate(self.get_E())]

        # column (E set) labels
        HEADER += '|'.join([str(e) + ' ' * (col_width[j] - len(str(e))) for j, e in enumerate(self.get_E())])
        result += HEADER + '\n'

        SEPARATOR = '-' * max_tabs + '+' + '+'.join(['-' * c for c in col_width])
        result += SEPARATOR + '\n'

        # print short words row labels
        for i, s_word in enumerate(self.get_S() + self.get_low_S()):
            if i == len(self.get_S()):
                result += SEPARATOR + '\n'
            row = rows[i]
            if filter_empty and not row.is_populated():
                pass
            else:
                ROW = str(s_word)
                ROW += ' ' * (max_tabs - len(str(s_word))) + '|'
                ROW += '|'.join([s.label + ' ' * (col_width[j] - len(s.label)) for j, s in enumerate(row.state)])
                result += ROW + '\n'

        result += SEPARATOR + '\n'
        return result

    def print(self, filter_empty=False):
        LOGGER.info(self.__str__(filter_empty))

    def get_loc_from_word(self, word: Trace, locations: List[Location], seq_to_loc: Dict[Trace, str], teacher):
        loc = None

        if word in seq_to_loc.keys():
            loc = [l for l in locations if l.name == seq_to_loc[word]][0]
        else:
            if word in self.get_S():
                curr_row = self.get_upper_observations()[self.get_S().index(word)]
            elif word in self.get_low_S():
                curr_row = self.get_lower_observations()[self.get_low_S().index(word)]
            elif Trace(word[:-1]) in self.get_S():
                row = self.get_S().index(Trace(word[:-1]))
                column = self.get_E().index(Trace([word[-1]]))
                needed_state = self.get_upper_observations()[row].state[column]
                curr_row = Row([needed_state] + [State([(None, None)])] * (len(self.get_E()) - 1))
            elif Trace(word[:-1]) in self.get_low_S():
                row = self.get_low_S().index(Trace(word[:-1]))
                column = self.get_E().index(Trace([word[-1]]))
                needed_state = self.get_lower_observations()[row].state[column]
                curr_row = Row([needed_state] + [State([(None, None)])] * (len(self.get_E()) - 1))

            if not curr_row.is_populated():
                return loc

            for i, row in enumerate(self.get_upper_observations()):
                if self.get_S()[i] in seq_to_loc.keys() and teacher.eqr_query(curr_row, row, strict=True):
                    loc = [l for l in locations if l.name == seq_to_loc[self.get_S()[i]]][0]
        return loc

    def add_init_edges(self, locations: List[Location], edges: List[Edge], seq_to_loc: Dict[Trace, str], teacher):
        init_loc = Location('__init__', None)
        locations.append(init_loc)

        one_word_upper = [word for word in self.get_S() if len(word) == 1]
        one_word_lower = [word for word in self.get_low_S() if len(word) == 1]

        for word in one_word_upper + one_word_lower:
            dest_loc = self.get_loc_from_word(word, locations, seq_to_loc, teacher)
            if dest_loc is not None:
                edges.append(Edge(init_loc, dest_loc, sync=str(word)))

        return locations, edges

    def to_sha(self, teacher):
        locations: List[Location] = []
        upp_obs: List[Row] = self.get_upper_observations()
        low_obs: List[Row] = self.get_lower_observations()
        # each unique sequence in the upper observations
        # constitutes an automaton location
        unique_sequences: List[Trace] = []
        unique_sequences_dict: Dict[Trace, str] = {}
        for i, row in enumerate(upp_obs):
            row_already_present = False
            for seq in unique_sequences:
                row_2 = upp_obs[self.get_S().index(seq)]
                if teacher.eqr_query(row, row_2, strict=True):
                    row_already_present = True
                    break
            if not row_already_present:
                unique_sequences.append(self.get_S()[i])
        # Create a new location for each unique sequence
        for index, seq in enumerate(unique_sequences):
            seq_index = self.get_S().index(seq)
            row = upp_obs[seq_index]
            new_name = StochasticHybridAutomaton.LOCATION_FORMATTER.format(len(locations))
            new_flow = row.state[0].vars[0][0].label + ', ' + row.state[0].vars[0][1].label
            locations.append(Location(new_name, new_flow))
            unique_sequences_dict[seq] = new_name

        # start building edges list for upper part of the table
        edges: List[Edge] = []
        for s_i, s_word in enumerate(self.get_S()):
            for t_i, t_word in enumerate(self.get_E()):
                if upp_obs[s_i].state[t_i].observed():
                    word: Trace = s_word + t_word
                    entry_word = Trace(word[:len(word) - 1])
                    if len(entry_word) == 0:
                        continue

                    start_loc = self.get_loc_from_word(entry_word, locations, unique_sequences_dict, teacher)
                    dest_loc = self.get_loc_from_word(word, locations, unique_sequences_dict, teacher)

                    labels = str(Trace(word[-1:]))
                    new_edge = Edge(start_loc, dest_loc, sync=labels)
                    if start_loc is not None and dest_loc is not None and new_edge not in edges:
                        edges.append(new_edge)

        # start building edges list for lower part of the table
        for s_i, s_word in enumerate(self.get_low_S()):
            for t_i, t_word in enumerate(self.get_E()):
                if low_obs[s_i].state[t_i].observed():
                    word: Trace = s_word + t_word
                    entry_word = Trace(word[:len(word) - 1])
                    if len(entry_word) == 0:
                        continue

                    start_loc = self.get_loc_from_word(entry_word, locations, unique_sequences_dict, teacher)
                    dest_loc = self.get_loc_from_word(word, locations, unique_sequences_dict, teacher)

                    if word != '':
                        labels = str(word.sub_prefix(entry_word))
                    else:
                        labels = EMPTY_STRING
                    new_edge = Edge(start_loc, dest_loc, sync=labels)
                    if start_loc is not None and dest_loc is not None and new_edge not in edges:
                        edges.append(new_edge)

        locations, edges = self.add_init_edges(locations, edges, unique_sequences_dict, teacher)

        return StochasticHybridAutomaton(locations, edges), unique_sequences_dict
