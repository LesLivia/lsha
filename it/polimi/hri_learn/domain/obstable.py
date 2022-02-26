from typing import List

from it.polimi.hri_learn.domain.lshafeatures import Trace, State


class Row:
    def __init__(self, state: List[State]):
        self.state = state

    def is_populated(self):
        return any([s.observed() for s in self.state])

    def __str__(self):
        return '\t|\t'.join([str(s) for s in self.state])

    def __eq__(self, other):
        return all([s == other.state[i] for i, s in enumerate(self.state)])


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
