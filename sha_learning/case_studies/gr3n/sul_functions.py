import configparser
import pandas as pd
import os
from typing import List
from datetime import datetime

from sha_learning.domain.lshafeatures import Event, FlowCondition, TimedTrace
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from sha_learning.learning_setup.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

try:
    CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))
except ValueError:
    CS_VERSION = None

LOGGER = Logger('SUL DATA HANDLER')
PUMP_SPEED_RANGE = int(config['GR3N']['PUMP_SPEED_RANGE'])
MIN_PUMP_SPEED = int(config['GR3N']['MIN_PUMP_SPEED'])
MAX_PUMP_SPEED = int(config['GR3N']['MAX_PUMP_SPEED'])

TMPRT_RANGE = int(config['GR3N']['TMPRT_RANGE'])
MIN_TMPRT = int(config['GR3N']['MIN_TMPRT'])
MAX_TMPRT = int(config['GR3N']['MAX_TMPRT'])


def is_chg_pt(curr, prev):
    for THRESHOLD in range(MIN_PUMP_SPEED, MAX_PUMP_SPEED, PUMP_SPEED_RANGE):
        if curr[0] < THRESHOLD <= prev[0] or prev[0] < THRESHOLD <= curr[0]:
            return True

    for THRESHOLD in range(MIN_TMPRT, MAX_TMPRT, TMPRT_RANGE):
        if curr[1] < THRESHOLD <= prev[1] or prev[1] < THRESHOLD <= curr[1]:
            return True

    return False

def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    pump_speed_sig = signals[1]
    pump_speed = {pt.timestamp: (i, pt.value) for i, pt in enumerate(pump_speed_sig.points)}

    tmprt_sig = signals[2]
    tmprt = {pt.timestamp: (i, pt.value) for i, pt in enumerate(tmprt_sig.points)}



    curr_pump_speed_index, curr_pump_speed = pump_speed[t]
    if curr_pump_speed_index > 0:
        try:
            prev_index = [tup[0] for tup in pump_speed.values() if tup[0] < curr_pump_speed_index][-1]
            prev_pump_speed = pump_speed_sig.points[prev_index].value
        except IndexError:
            prev_pump_speed = None
    else:
        prev_pump_speed = curr_pump_speed

    curr_tmprt_index, curr_tmprt = tmprt[t]
    if curr_tmprt_index > 0:
        try:
            prev_index = [tup[0] for tup in tmprt.values() if tup[0] < curr_tmprt_index][-1]
            prev_tmprt = tmprt_sig.points[prev_index].value
        except IndexError:
            prev_tmprt = None
    else:
        prev_tmprt = curr_tmprt



    identified_event = None
    if prev_tmprt is not None: # for now we just ignore prev_tmprt None, but in case this function have to be revised
        # Identify event as in is_chg_pts
        for i, THRESHOLD in enumerate(range(MIN_TMPRT, MAX_TMPRT, TMPRT_RANGE)):
            if curr_tmprt < THRESHOLD <= prev_tmprt or prev_tmprt < THRESHOLD <= curr_tmprt:
                identified_event = events[i + int((MAX_PUMP_SPEED - MIN_PUMP_SPEED)/PUMP_SPEED_RANGE)]
    else:
        identified_event = events[int((MAX_PUMP_SPEED - MIN_PUMP_SPEED)/PUMP_SPEED_RANGE)]

    if prev_pump_speed is not None:  # for now we just ignore prev_tmprt None, but in case this function have to be revised
        for i, THRESHOLD in enumerate(range(MIN_PUMP_SPEED, MAX_PUMP_SPEED, PUMP_SPEED_RANGE)):
            if curr_pump_speed < THRESHOLD <= prev_pump_speed or prev_pump_speed < THRESHOLD <= curr_pump_speed:
                identified_event = events[i]  # I already know that there is an event for the pump speed
    else:
        identified_event = events[0]



    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: datetime):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


def parse_data(path: str):
    pump_speed: SampledSignal = SampledSignal([], label='sp')
    Talim: SampledSignal = SampledSignal([], label='Ta')
    tmprt: SampledSignal = SampledSignal([], label='tmp')

    dd_real = pd.read_csv(path)

    dd_pump_speed = dd_real[dd_real['DataObjectField'] == 'SpeedSP']
    dd_pump_speed.loc[:, 'TimeStamp'] = pd.to_datetime(dd_pump_speed['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_pump_speed.sort_values(by='TimeStamp')

    dd_Talim = dd_real[dd_real['DataObjectField'] == 'TCuscinettiAlimentazione']
    dd_Talim.loc[:, 'TimeStamp'] = pd.to_datetime(dd_Talim['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_Talim.sort_values(by='TimeStamp')

    dd_tmprt = dd_real[dd_real['DataObjectField'] == 'Value']
    dd_tmprt.loc[:, 'TimeStamp'] = pd.to_datetime(dd_tmprt['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_tmprt.sort_values(by='TimeStamp')

    pump_speed.points.extend([SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_pump_speed.iterrows()])
    Talim.points.extend([SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_Talim.iterrows()])
    tmprt.points.extend([SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_tmprt.iterrows()])

    return [Talim, pump_speed, tmprt]


def get_absorption_param(segment: List[SignalPoint], flow: FlowCondition):
    if len(segment) != 0:
        sum_abs = sum([pt.value for pt in segment])
        avg_abs = sum_abs / (len(segment))
        return avg_abs

    return 0