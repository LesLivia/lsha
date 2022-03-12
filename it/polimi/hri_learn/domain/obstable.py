from typing import List, Dict

from it.polimi.hri_learn.domain.hafeatures import HybridAutomaton, Location, Edge
from it.polimi.hri_learn.domain.lshafeatures import Trace, State, EMPTY_STRING


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
        try:
            max_s = max([len(word) for word in self.get_S()])
        except ValueError:
            max_s = 0
        try:
            max_low_s = max([len(word) for word in self.get_low_S()])
        except ValueError:
            max_low_s = 0
        max_tabs = int(max(max_s, max_low_s))

        HEADER = '\t' * max_tabs + '|\t\t'

        # print column labels
        for t_word in self.get_E():
            HEADER += str(t_word) + '\t\t|\t\t'
        result += HEADER + '\n'

        SEPARATOR = '----' * max_tabs + '+' + '---------------+' * len(self.get_E())
        result += SEPARATOR + '\n'

        # print short words row labels
        for (i, s_word) in enumerate(self.get_S()):
            row = self.get_upper_observations()[i]
            if filter_empty and not row.is_populated():
                pass
            else:
                ROW = str(s_word)
                len_word = int(len(s_word) / 3) if s_word != '' else 1
                ROW += '\t' * (max_tabs + 1 - len_word) + '|\t' if len_word < max_tabs - 1 or max_tabs <= 4 \
                    else '\t' * (max_tabs + 2 - len_word) + '|\t'
                ROW += str(row) + '\n'
                result += ROW
        result += SEPARATOR + '\n'

        # print long words row labels
        for (i, s_word) in enumerate(self.get_low_S()):
            row = self.get_lower_observations()[i]
            if filter_empty and not row.is_populated():
                pass
            else:
                ROW = str(s_word)
                len_word = int(len(s_word) / 3)
                ROW += '\t' * (max_tabs + 1 - len_word) + '|\t' if len_word < max_tabs - 1 or max_tabs <= 4 \
                    else '\t' * (max_tabs + 2 - len_word) + '|\t'
                ROW += str(row) + '\n'
                result += ROW
        result += SEPARATOR + '\n'
        return result

    def print(self, filter_empty=False):
        print(self.__str__(filter_empty))

    def get_loc_from_word(self, word: Trace, locations: List[Location], seq_to_loc: Dict[Trace, str]):
        loc = None
        if word in seq_to_loc.keys():
            loc = [l for l in locations if l.name == seq_to_loc[word]][0]
        else:
            try:
                curr_row = self.get_upper_observations()[self.get_S().index(word)]
            except ValueError:
                curr_row = self.get_lower_observations()[self.get_low_S().index(word)]

            for i, row in enumerate(self.get_upper_observations()):
                if self.get_S()[i] in seq_to_loc.keys() and curr_row == row:
                    loc = [l for l in locations if l.name == seq_to_loc[self.get_S()[i]]][0]
        return loc

    def add_init_edges(self, locations: List[Location], edges: List[Edge], seq_to_loc: Dict[Trace, str]):
        init_loc = Location('__init__', 'null')
        locations.append(init_loc)

        one_word_upper = [word for word in self.get_S() if len(word) == 1]
        # TODO
        one_word_lower = [word for word in self.get_low_S() if len(word) == 1]

        for word in one_word_upper:
            dest_loc = self.get_loc_from_word(word, locations, seq_to_loc)
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
                if teacher.eqr_query(row, row_2):
                    row_already_present = True
                    break
            if not row_already_present:
                unique_sequences.append(self.get_S()[i])
        # Create a new location for each unique sequence
        for index, seq in enumerate(unique_sequences):
            seq_index = self.get_S().index(seq)
            row = upp_obs[seq_index]
            new_name = HybridAutomaton.LOCATION_FORMATTER.format(seq_index)
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

                    start_loc = self.get_loc_from_word(entry_word, locations, unique_sequences_dict)
                    dest_loc = self.get_loc_from_word(word, locations, unique_sequences_dict)

                    labels = str(Trace(word[-1:]))
                    new_edge = Edge(start_loc, dest_loc, sync=labels)
                    if new_edge not in edges:
                        edges.append(new_edge)

        # start building edges list for lower part of the table
        for s_i, s_word in enumerate(self.get_low_S()):
            for t_i, t_word in enumerate(self.get_E()):
                if low_obs[s_i].state[t_i].observed():
                    word: Trace = s_word + t_word
                    entry_word = Trace(word[:len(word) - 1])
                    if len(entry_word) == 0:
                        continue

                    start_loc = self.get_loc_from_word(entry_word, locations, unique_sequences_dict)
                    dest_loc = self.get_loc_from_word(word, locations, unique_sequences_dict)

                    if word != '':
                        labels = str(word).replace(str(entry_word), '')
                    else:
                        labels = EMPTY_STRING
                    new_edge = Edge(start_loc, dest_loc, sync=labels)
                    if new_edge not in edges:
                        edges.append(new_edge)

        locations, edges = self.add_init_edges(locations, edges, unique_sequences_dict)

        return HybridAutomaton(locations, edges)
