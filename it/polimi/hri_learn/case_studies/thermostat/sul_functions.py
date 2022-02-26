import configparser
import math
from typing import List

from it.polimi.hri_learn.domain.lshafeatures import FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp, Event, SignalPoint
from it.polimi.hri_learn.lstar_sha.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS_VERSION = config['SUL CONFIGURATION']['CS_VERSION'][0]
ON_R = 100.0
LOGGER = Logger('SUL DATA HANDLER')


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    try:
        wOpen = signals[2]
    except IndexError:
        wOpen: SampledSignal = SampledSignal([])
    heatOn = signals[0]
    t = t.to_secs()

    identified_guard = ''
    curr_wOpen = list(filter(lambda x: x.timestamp.to_secs() <= t, wOpen.points))[-1]
    if CS_VERSION == 'a':
        identified_guard += events[0].guard if curr_wOpen.value == 1.0 else '!' + events[0].guard
    if CS_VERSION == 'b' or CS_VERSION == 'c':
        identified_guard += events[1].guard if curr_wOpen.value == 2.0 else '!' + events[1].guard
        # identified_guard += self.get_guards()[2] if curr_wOpen.value == 0.0 else '!' + self.get_guards()[2]

    curr_heatOn = list(filter(lambda x: x.timestamp.to_secs() <= t, heatOn.points))[-1]
    identified_channel = events[0].chan if curr_heatOn.value == 1.0 else events[1].chan

    identified_event = [e for e in events if e.guard == identified_guard and e.chan == identified_channel][0]
    return identified_event


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    f = open(path, 'r')
    variables = ['t.ON', 'T_r', 'r.open']
    lines = f.readlines()
    split_indexes = [lines.index('# ' + k + ' #1\n') for k in variables]
    split_lines = [lines[i + 1:split_indexes[ind + 1]] for (ind, i) in enumerate(split_indexes) if
                   i != split_indexes[-1]]
    split_lines.append(lines[split_indexes[-1] + 1:len(lines)])
    new_signals: List[SampledSignal] = []
    for (i, v) in enumerate(variables):
        entries = [line.split(' ') for line in split_lines[i]][1:]
        t = [float(x[0]) for x in entries]
        values = [float(x[1]) for x in entries]
        ts = [Timestamp(0, 0, 0, 0, 0, i) for i in t]
        signal_pts: List[SignalPoint] = [SignalPoint(ts[i], values[i]) for i in range(len(ts))]
        new_signal: SampledSignal = SampledSignal(signal_pts, v)
        new_signals.append(new_signal)
    return new_signals


def get_thermo_param(segment: List[SignalPoint], flow: FlowCondition):
    try:
        val = [pt.value for pt in segment]
        if flow.f_id in [0, 2]:
            if CS_VERSION != 'c' or (CS_VERSION == 'c' and flow.f_id == 0):
                increments = []
                for (i, pt) in enumerate(val):
                    if i > 0 and pt != val[i - 1]:
                        increments.append(pt - val[i - 1] * math.exp(-1 / ON_R))
                Ks = [delta_t / (ON_R * (1 - math.exp(-1 / ON_R))) for delta_t in increments if delta_t != 0]

                LOGGER.info('Estimating rate with heat on ({})'.format(flow.f_id))
                est_rate = sum(Ks) / len(Ks) if len(Ks) > 0 else None
            else:
                increments = []
                for (i, pt) in enumerate(val):
                    if i > 0:
                        increments.append(pt - val[i - 1])
                increments = [i for i in increments if i != 0]
                LOGGER.info('Estimating rate with heat on ({})'.format(flow.f_id))
                est_rate = sum(increments) / len(increments) if len(increments) > 0 else None
        else:
            increments = []
            for (i, pt) in enumerate(val):
                if i > 0:
                    increments.append(pt / val[i - 1])

            Rs = [-1 / math.log(delta_t) for delta_t in increments if delta_t != 1]

            LOGGER.info('Estimating rate with heat off ({})'.format(flow.f_id))
            est_rate = sum(Rs) / len(Rs) if len(Rs) > 0 else None

        return est_rate
    except ValueError:
        return None
