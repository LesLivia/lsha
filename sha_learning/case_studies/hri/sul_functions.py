import configparser
import math
import os
from typing import List

from sha_learning.domain.lshafeatures import FlowCondition
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp, Event, SignalPoint
from sha_learning.learning_setup.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', '')[0])
SAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY']
LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    return curr != prev


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    posX = signals[1]
    moving = signals[2]

    '''
    Repeat for every channel in the system
    '''
    curr_mov = list(filter(lambda x: x.timestamp == t, moving.points))[0]
    identified_channel = 'start' if curr_mov.value == 1 else 'stop'

    identified_guard = ''
    if (SAMPLE_STRATEGY == 'SIM' and CS_VERSION in [1, 2, 3]) or \
            (SAMPLE_STRATEGY == 'UPPAAL' and CS_VERSION in [2, 3, 4, 5]):
        posY = signals[3]
        curr_posx = list(filter(lambda x: x.timestamp <= t, posX.points))[-1]
        curr_posy = list(filter(lambda x: x.timestamp <= t, posY.points))[-1]
        if SAMPLE_STRATEGY == 'UPPAAL':
            in_waiting = curr_posx.value >= 2000.0 and curr_posy.value <= 3000.0
        else:
            in_waiting = 16 <= curr_posx.value <= 23.0 and 1.0 <= curr_posy.value <= 10.0
        identified_guard += 'sit' if in_waiting else '!sit'

    if (SAMPLE_STRATEGY == 'SIM' and CS_VERSION in [1, 2, 3]) or \
            (SAMPLE_STRATEGY == 'UPPAAL' and CS_VERSION in [3, 4, 5]):
        posY = signals[3]
        curr_posx = list(filter(lambda x: x.timestamp <= t, posX.points))[-1]
        curr_posy = list(filter(lambda x: x.timestamp <= t, posY.points))[-1]
        if SAMPLE_STRATEGY == 'UPPAAL':
            in_office = curr_posx.value >= 2000.0 and 1000.0 <= curr_posy.value <= 3000.0
        else:
            in_office = 1.0 <= curr_posx.value <= 11.0 and 1.0 <= curr_posy.value <= 10.0
        identified_guard += 'run' if in_office else '!run'

    if SAMPLE_STRATEGY == 'SIM' and CS_VERSION in [4]:
        posY = signals[3]
        curr_posx = list(filter(lambda x: x.timestamp <= t, posX.points))[-1]
        curr_posy = list(filter(lambda x: x.timestamp <= t, posY.points))[-1]

        close_to_chair = 16 <= curr_posx.value <= 20.0 and 3.0 <= curr_posy.value <= 6.0
        identified_guard += 's' if close_to_chair and identified_channel == 'stop' else '!s'

        next_sig_x = list(filter(lambda x: x.timestamp > t, posX.points))
        next_sig_y = list(filter(lambda x: x.timestamp > t, posY.points))
        nextx = next_sig_x[0] if len(next_sig_x) > 0 else curr_posx
        nexty = next_sig_y[0] if len(next_sig_y) > 0 else curr_posy
        dist = math.sqrt((nextx.value - curr_posx.value) ** 2 + (nexty.value - curr_posy.value) ** 2)
        vel = dist / 2.0

        room = signals[4]
        curr_room_status = list(filter(lambda x: x.timestamp <= t.to_secs(), room.points))[-1]
        identified_guard += 'r' if vel > 0.8 and ((close_to_chair and identified_channel == 'stop')
                                                  or curr_room_status.value) else '!r'
        identified_guard += 'h' if curr_room_status.value and not (close_to_chair and vel > 1.0) else '!h'

        identified_guard += 'l' if 0.2 <= vel < 0.4 and not close_to_chair \
                                   and (identified_channel == 'start' or curr_room_status.value) else '!l'
        identified_guard += 'a' if vel < 0.2 and not curr_room_status else '!a'

    '''
    Find symbol associated with guard-channel combination
    '''
    identified_event = [e for e in events if e.guard == identified_guard and e.chan == identified_channel][0]
    return identified_event


def parse_data(path: str):
    if SAMPLE_STRATEGY == 'SIM':
        return parse_traces_sim(path)
    else:
        return parse_traces_uppaal(path)


def parse_traces_sim(path: str):
    if CS_VERSION == 4:
        logs = ['humanFatigue.log', 'humanPosition.log', 'environmentData.log']
    else:
        logs = ['humanFatigue.log', 'humanPosition.log']

    new_traces: List[SampledSignal] = []
    for i, log in enumerate(logs):
        f = open(path + log)
        lines = f.readlines()[1:]
        lines = [line.replace('\n', '') for line in lines]
        t = [float(line.split(':')[0]) for line in lines]
        if i == 0:
            v = [float(line.split(':')[2]) for line in lines]
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, x), v[j]) for j, x in enumerate(t)]
            new_traces.append(SampledSignal(signal, str(len(new_traces))))
        elif i == 1:
            pos = [line.split(':')[2] for line in lines]
            pos_x = [float(line.split('#')[0]) for line in pos]
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, x), pos_x[j]) for j, x in enumerate(t)]
            new_traces.append(SampledSignal(signal, 'humanPositionX'))
            busy = [float(v != pos[j - 1]) for j, v in enumerate(pos) if j > 0]
            busy = [0.0] + busy
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, x), busy[j]) for j, x in enumerate(t)]
            new_traces.append(SampledSignal(signal, str(len(new_traces))))
            pos_y = [float(line.split('#')[1]) for line in pos]
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, x), pos_y[j]) for j, x in enumerate(t)]
            new_traces.append(SampledSignal(signal, 'humanPositionY'))
        else:
            data = [line.split(':')[1] for line in lines]
            data = list(map(lambda x: (float(x.split('#')[0]), float(x.split('#')[1])), data))
            harsh = []
            for pt in data:
                temp = pt[0]
                hum = pt[1]
                harsh.append(temp <= 12.0 or temp >= 32.0 or hum <= 30.0 or hum >= 60.0)
            signal = [SignalPoint(x, harsh[j]) for (j, x) in enumerate(t)]
            new_traces.append(SampledSignal(signal, log))

    return new_traces


def parse_traces_uppaal(path: str):
    f = open(path, 'r')
    if CS_VERSION in [1, 2]:
        variables = ['humanFatigue[currH - 1]', 'humanPositionX[currH - 1]',
                     'amy.busy || amy.p_2', 'humanPositionY[currH - 1]']
    else:
        variables = ['humanFatigue[currH - 1]', 'humanPositionX[currH - 1]',
                     'amy.busy || amy.p_2 || amy.run || amy.p_4', 'humanPositionY[currH - 1]']
    lines = f.readlines()
    split_indexes = [lines.index(k + ':\n') for k in variables]
    split_lines = [lines[i + 1:split_indexes[ind + 1]] for (ind, i) in enumerate(split_indexes) if
                   i != split_indexes[-1]]
    split_lines.append(lines[split_indexes[-1] + 1:len(lines)])
    traces = len(split_lines[0])
    new_traces: List[SampledSignal] = []
    for trace in range(traces):
        for i, v in enumerate(variables):
            entries = split_lines[i][trace].split(' ')
            entries = entries[1:]
            for e in entries:
                new = e.replace('(', '')
                new = new.replace(')', '')
                entries[entries.index(e)] = new
            t = [float(x.split(',')[0]) for x in entries]
            v = [float(x.split(',')[1]) for x in entries]
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, t[i]), v[i]) for i in range(len(t))]
            new_traces.append(SampledSignal(signal, str(i)))
    return new_traces


def get_ftg_param(segment: List[SignalPoint], flow: FlowCondition):
    try:
        val = [pt.value for pt in segment]
        # metric for walking
        if flow.f_id == 1:
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
