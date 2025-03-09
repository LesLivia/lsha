import configparser
import csv
import pandas as pd
import os
from typing import List, Tuple, Dict
from datetime import datetime

from sha_learning.domain.lshafeatures import Event, FlowCondition
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

COPPIA_MIDPOINT = int(config['GR3N']['COPPIA_MIDPOINT'])
LOGGER = Logger('SUL DATA HANDLER')

def is_chg_pt(curr, prev):
    return  (curr[0] > COPPIA_MIDPOINT and prev[0] < COPPIA_MIDPOINT) or \
            (curr[0] < COPPIA_MIDPOINT and prev[0] > COPPIA_MIDPOINT)

def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    coppia_sig = signals[1]
    coppia = {pt.timestamp: (i, pt.value) for i, pt in enumerate(coppia_sig.points)}

    curr_coppia_index, curr_coppia = coppia[t]
    if curr_coppia_index > 0:
        try:
            prev_index = [tup[0] for tup in coppia.values() if tup[0] < curr_coppia_index][-1]
            prev_coppia = coppia_sig.points[prev_index].value
        except IndexError:
            prev_coppia = None
    else:
        prev_coppia = curr_coppia

    identified_event = None
    if (curr_coppia > COPPIA_MIDPOINT and prev_coppia < COPPIA_MIDPOINT):
        identified_event = events[0]
    elif (curr_coppia < COPPIA_MIDPOINT and prev_coppia > COPPIA_MIDPOINT):
        identified_event = events[1]

    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: str):
    date = ts.split(' ')[0].split('-')
    time = ts.split(' ')[1].split(':')
    return Timestamp(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), int(time[2].split('.')[0]))


def parse_data(path: str):
    differenziale: SampledSignal = SampledSignal([], label='d')
    assorbimento: SampledSignal = SampledSignal([], label='a')
    coppia: SampledSignal = SampledSignal([], label='cp')

    dd_real = pd.read_csv('D:\\Uni\\Magistrale\\1 Anno\\1 semestre\\Software engineering 2\\Gr3n\\csv\\20250202_DecanterData_REAL.csv')

    dd_differenziale = dd_real[dd_real['DataObjectField'] == 'Differenziale']
    dd_differenziale.sort_values(by='time')

    dd_assorbimento = dd_real[dd_real['DataObjectField'] == 'Assorbimento']
    dd_assorbimento.sort_values(by='time')

    dd_coppia = dd_real[dd_real['DataObjectField'] == 'Coppia']
    dd_coppia.sort_values(by='time')

    differenziale.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_differenziale.iterrows()])
    assorbimento.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_assorbimento.iterrows()])
    coppia.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_coppia.iterrows()])

    return [assorbimento, coppia, differenziale]


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_power = sum([pt.value for pt in segment])
    avg_power = sum_power / len(segment)
    return avg_power
