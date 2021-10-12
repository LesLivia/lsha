import configparser
import math
import sys
from typing import List

from it.polimi.hri_learn.domain.sigfeatures import SignalPoint
from it.polimi.hri_learn.lstar_sha.logger import Logger

'''
WARNING! 
        These constants may change if the system changes:
        default model and distr. for empty string.
'''
LOGGER = Logger()
config = configparser.ConfigParser()
config.sections()
config.read(sys.argv[1])
config.sections()

CASE_STUDY = config['SUL CONFIGURATION']['CASE_STUDY']
CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'])
MAIN_SIGNAL = None

if CASE_STUDY == 'HRI':
    MAIN_SIGNAL = 0
    DRIVER_SIG = 2
    DEFAULT_MODEL = 0
    DEFAULT_DISTR = 0
    MODEL_TO_DISTR_MAP = {0: 0, 1: 1}  # <- HRI
    # MODEL_TO_DISTR_MAP = {0: 0, 2: 0, 4: 0, 6: 0, 8: 0, 1: 1, 3: 1, 5: 1, 7: 1, 9: 1}
else:
    ON_R = 100.0
    MAIN_SIGNAL = 1
    DRIVER_SIG = 0
    if CS_VERSION == 'a' or CS_VERSION == 'b':
        MODEL_TO_DISTR_MAP = {0: 0, 1: 1}  # <- THERMOSTAT
        DEFAULT_MODEL = 0
        DEFAULT_DISTR = 0
    else:
        MODEL_TO_DISTR_MAP = {0: 0, 1: 2}
        DEFAULT_MODEL = 0
        DEFAULT_DISTR = 0


class EventFactory:
    def __init__(self, guards, channels, symbols):
        self.guards = guards
        self.channels = channels
        self.symbols = symbols
        self.signals: List[List[List[SignalPoint]]] = []

    def clear(self):
        self.signals.append([])

    def set_guards(self, guards):
        self.guards = guards

    def set_channels(self, channels):
        self.channels = channels

    def set_symbols(self, symbols):
        self.symbols = symbols

    def add_signal(self, signal, trace):
        self.signals[trace].append(signal)

    def get_guards(self):
        return self.guards

    def get_channels(self):
        return self.channels

    def get_symbols(self):
        return self.symbols

    def get_signals(self):
        return self.signals

    '''
    WARNING! 
            This method must be RE-IMPLEMENTED for each system:
            each guard corresponds to a specific condition on a signal,
            the same stands for channels.
    '''

    def label_event(self, timestamp: float, trace):
        if CASE_STUDY=='HRI':
            posX = self.get_signals()[trace][1]
            moving = self.get_signals()[trace][2]

            identified_guard = ''
            '''
            Repeat for every guard in the system
            '''
            if CS_VERSION in [1, 2]:
                posY = self.get_signals()[trace][3]
                curr_posx = list(filter(lambda x: x.timestamp <= timestamp, posX))[-1]
                curr_posy = list(filter(lambda x: x.timestamp <= timestamp, posY))[-1]
                # in_waiting = curr_posx.value >= 2000.0 and curr_posy.value <= 3000.0
                in_waiting = 16 <= curr_posx.value <= 23.0 and 1.0 <= curr_posy.value <= 10.0
                identified_guard += self.get_guards()[0] if in_waiting else '!' + self.get_guards()[0]

            if CS_VERSION in ['c']:
                posY = self.get_signals()[trace][3]
                curr_posx = list(filter(lambda x: x.timestamp <= timestamp, posX))[-1]
                curr_posy = list(filter(lambda x: x.timestamp <= timestamp, posY))[-1]
                # in_office = curr_posx.value >= 2000.0 and 1000.0 <= curr_posy.value <= 3000.0
                in_office = 1.0 <= curr_posx.value <= 11.0 and 1.0 <= curr_posy.value <= 10.0
                identified_guard += self.get_guards()[1] if in_office else '!' + self.get_guards()[1]

            if CS_VERSION in ['x']:
                posY = self.get_signals()[trace][3]
                curr_posx = list(filter(lambda x: x.timestamp <= timestamp, posX))[-1]
                curr_posy = list(filter(lambda x: x.timestamp <= timestamp, posY))[-1]

                close_to_chair = 16 <= curr_posx.value <= 20.0 and 3.0 <= curr_posy.value <= 6.0
                identified_guard += self.get_guards()[0] if close_to_chair else '!' + self.get_guards()[0]

                next_sig_x = list(filter(lambda x: x.timestamp > timestamp, posX))
                next_sig_y = list(filter(lambda x: x.timestamp > timestamp, posY))
                nextx = next_sig_x[0] if len(next_sig_x) > 0 else curr_posx
                nexty = next_sig_y[0] if len(next_sig_y) > 0 else curr_posy
                dist = math.sqrt((nextx.value - curr_posx.value) ** 2 + (nexty.value - curr_posy.value) ** 2)
                vel = dist / 2.0
                identified_guard += self.get_guards()[1] if vel > 1.0 else '!' + self.get_guards()[1]

                room = self.get_signals()[trace][4]
                curr_room_status = list(filter(lambda x: x.timestamp <= timestamp, room))[-1]
                identified_guard += self.get_guards()[2] if curr_room_status else '!' + self.get_guards()[2]

                identified_guard += self.get_guards()[3] if vel < 0.2 else '!' + self.get_guards()[3]
                identified_guard += self.get_guards()[4] if False else '!' + self.get_guards()[4]

            '''
            Repeat for every channel in the system
            '''
            curr_mov = list(filter(lambda x: x.timestamp == timestamp, moving))[0]
            identified_channel = self.get_channels()[0] if curr_mov.value == 1 else self.get_channels()[1]
        else:
            wOpen = self.get_signals()[trace][2]
            heatOn = self.get_signals()[trace][0]

            identified_guard = ''
            '''
            Repeat for every guard in the system
            '''
            curr_wOpen = list(filter(lambda x: x.timestamp <= timestamp, wOpen))[-1]
            identified_guard += self.get_guards()[0] if curr_wOpen.value == 1.0 else '!' + self.get_guards()[0]
            if CS_VERSION == 'b' or CS_VERSION == 'c':
                identified_guard += self.get_guards()[1] if curr_wOpen.value == 2.0 else '!' + self.get_guards()[1]
                # identified_guard += self.get_guards()[2] if curr_wOpen.value == 0.0 else '!' + self.get_guards()[2]

            '''
            Repeat for every channel in the system
            '''
            curr_heatOn = list(filter(lambda x: x.timestamp == timestamp, heatOn))[0]
            identified_channel = self.get_channels()[0] if curr_heatOn.value == 1.0 else self.get_channels()[1]

        '''
        Find symbol associated with guard-channel combination
        '''
        combination = identified_guard + ' and ' + identified_channel
        for key in self.get_symbols().keys():
            if self.get_symbols()[key] == combination:
                return key

    '''
    WARNING! 
            This method must be RE-IMPLEMENTED for each system:
            returns metric for HT queries.
    '''

    def get_ht_param(self, segment: List[SignalPoint], model=None):
        if CASE_STUDY == 'HRI':
            return self.get_ftg_param(segment, model)
        else:
            return self.get_thermo_param(segment, model)

    def get_ftg_param(self, segment: List[SignalPoint], model: int):
        try:
            val = [pt.value for pt in segment]
            # metric for walking
            if model == 1:
                lambdas = []
                for (i, v) in enumerate(val):
                    if i > 0 and v != val[i - 1]:
                        lambdas.append(math.log((1 - v) / (1 - val[i - 1])))
                est_rate = sum(lambdas) / len(lambdas) if len(lambdas) > 0 else None
            # metric for standing/sitting
            else:
                mus = []
                for (i, v) in enumerate(val):
                    if i > 0 and v != val[i - 1] and val[i - 1] != 0:
                        mus.append(math.log(v / val[i - 1]))
                est_rate = sum(mus) / len(mus) if len(mus) > 0 else None

            return abs(est_rate) if est_rate is not None else None
        except ValueError:
            return None

    def get_thermo_param(self, segment: List[SignalPoint], model: int):
        try:
            val = [pt.value for pt in segment]
            if model in [1, 2]:
                if CS_VERSION != 'c' or (CS_VERSION == 'c' and model == 1):
                    increments = []
                    for (i, pt) in enumerate(val):
                        if i > 0 and pt != val[i - 1]:
                            increments.append(pt - val[i - 1] * math.exp(-1 / ON_R))
                    Ks = [delta_t / (ON_R * (1 - math.exp(-1 / ON_R))) for delta_t in increments if delta_t != 0]

                    LOGGER.info('Estimating rate with heat on ({})'.format(model))
                    est_rate = sum(Ks) / len(Ks) if len(Ks) > 0 else None
                else:
                    increments = []
                    for (i, pt) in enumerate(val):
                        if i > 0:
                            increments.append(pt - val[i - 1])
                    increments = [i for i in increments if i != 0]
                    LOGGER.info('Estimating rate with heat on ({})'.format(model))
                    est_rate = sum(increments) / len(increments) if len(increments) > 0 else None
            else:
                increments = []
                for (i, pt) in enumerate(val):
                    if i > 0:
                        increments.append(pt / val[i - 1])

                Rs = [-1 / math.log(delta_t) for delta_t in increments if delta_t != 1]

                LOGGER.info('Estimating rate with heat off ({})'.format(model))
                est_rate = sum(Rs) / len(Rs) if len(Rs) > 0 else None

            return est_rate
        except ValueError:
            return None

    def parse_traces(self, path: str):
        if CASE_STUDY == 'hri_sim':
            return self.parse_traces_sim(path)
        else:
            return self.parse_traces_uppaal(path)

    def parse_traces_sim(self, path: str):
        if CS_VERSION == 'x':
            logs = ['humanFatigue.log', 'humanPosition.log', 'environmentData.log']
        else:
            logs = ['humanFatigue.log', 'humanPosition.log']

        new_traces = [[]]
        for (i, log) in enumerate(logs):
            f = open(path + log)
            lines = f.readlines()[1:]
            lines = [line.replace('\n', '') for line in lines]
            t = [float(line.split(':')[0]) for line in lines]
            if i == 0:
                v = [float(line.split(':')[2]) for line in lines]
                signal = [SignalPoint(x, 0, v[j]) for (j, x) in enumerate(t)]
                new_traces[0].append(signal)
            elif i == 1:
                pos = [line.split(':')[2] for line in lines]
                pos_x = [float(line.split('#')[0]) for line in pos]
                new_traces[0].append([SignalPoint(x, 0, pos_x[j]) for (j, x) in enumerate(t)])
                busy = [float(v != pos[j - 1]) for (j, v) in enumerate(pos) if j > 0]
                busy = [0.0] + busy
                new_traces[0].append([SignalPoint(x, 0, busy[j]) for (j, x) in enumerate(t)])
                pos_y = [float(line.split('#')[1]) for line in pos]
                new_traces[0].append([SignalPoint(x, 0, pos_y[j]) for (j, x) in enumerate(t)])
            else:
                data = [line.split(':')[1] for line in lines]
                data = list(map(lambda x: (float(x.split('#')[0]), float(x.split('#')[1])), data))
                harsh = []
                for pt in data:
                    temp = pt[0]
                    hum = pt[1]
                    harsh.append(temp <= 12.0 or temp >= 32.0 or hum <= 30.0 or hum >= 60.0)
                new_traces[0].append([SignalPoint(x, 0, harsh[j]) for (j, x) in enumerate(t)])

        return new_traces

    def parse_traces_uppaal(self, path: str):
        # support method to parse traces sampled by ref query
        f = open(path, 'r')
        if CASE_STUDY == 'HRI':
            if CS_VERSION in [1, 2]:
                variables = ['humanFatigue[currH - 1]', 'humanPositionX[currH - 1]',
                             'amy.busy || amy.p_2', 'humanPositionY[currH - 1]']
            else:
                variables = ['humanFatigue[currH - 1]', 'humanPositionX[currH - 1]',
                             'amy.busy || amy.p_2 || amy.run || amy.p_4', 'humanPositionY[currH - 1]']
        else:
            variables = ['t.ON', 'T_r', 'r.open']
        lines = f.readlines()
        split_indexes = [lines.index(k + ':\n') for k in variables]
        split_lines = [lines[i + 1:split_indexes[ind + 1]] for (ind, i) in enumerate(split_indexes) if
                       i != split_indexes[-1]]
        split_lines.append(lines[split_indexes[-1] + 1:len(lines)])
        traces = len(split_lines[0])
        new_traces = []
        for trace in range(traces):
            new_traces.append([])
            for (i, v) in enumerate(variables):
                entries = split_lines[i][trace].split(' ')
                entries = entries[1:]
                for e in entries:
                    new = e.replace('(', '')
                    new = new.replace(')', '')
                    entries[entries.index(e)] = new
                t = [float(x.split(',')[0]) for x in entries]
                v = [float(x.split(',')[1]) for x in entries]
                signal = [SignalPoint(t[i], 1, v[i]) for i in range(len(t))]
                new_traces[-1].append(signal)
        return new_traces
