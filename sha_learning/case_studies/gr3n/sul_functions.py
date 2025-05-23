import configparser
import pandas as pd
import os
from typing import List, Tuple, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

'''
TALIM_RANGE = int(config['GR3N']['TALIM_RANGE'])
MIN_TALIM = int(config['GR3N']['MIN_TALIM'])
MAX_TALIM = int(config['GR3N']['MAX_TALIM'])
'''

TRIDU_RANGE = int(config['GR3N']['TRIDU_RANGE'])
MIN_TRIDU = int(config['GR3N']['MIN_TRIDU'])
MAX_TRIDU = int(config['GR3N']['MAX_TRIDU'])

TMPRT_RANGE = int(config['GR3N']['TMPRT_RANGE'])
MIN_TMPRT = int(config['GR3N']['MIN_TMPRT'])
MAX_TMPRT = int(config['GR3N']['MAX_TMPRT'])


def is_chg_pt(curr, prev):
    for THRESHOLD in range(MIN_PUMP_SPEED, MAX_PUMP_SPEED, PUMP_SPEED_RANGE):
        if curr[0] < THRESHOLD <= prev[0] or prev[0] < THRESHOLD <= curr[0]:
            return True # I already know that there is an event for the pump speed

    for THRESHOLD in range(MIN_TRIDU, MAX_TRIDU, TRIDU_RANGE):
        if curr[1] < THRESHOLD <= prev[1] or prev[1] < THRESHOLD <= curr[1]:
            return True # I already know that there is an event for the TriduttoreCuscinetti

    for THRESHOLD in range(MIN_TMPRT, MAX_TMPRT, TMPRT_RANGE):
        if curr[2] < THRESHOLD <= prev[2] or prev[2] < THRESHOLD <= curr[2]:
            return True # I already know that there is an event for the temperature

    return False

def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    pump_speed_sig = signals[1]
    pump_speed = {pt.timestamp: (i, pt.value) for i, pt in enumerate(pump_speed_sig.points)}

    Tridu_sig = signals[2]
    Tridu = {pt.timestamp: (i, pt.value) for i, pt in enumerate(Tridu_sig.points)}

    tmprt_sig = signals[3]
    tmprt = {pt.timestamp: (i, pt.value) for i, pt in enumerate(tmprt_sig.points)}

    '''
    COPPIA_INTERVALS: List[Tuple[int, int]] = []
    for i in range(MIN_COPPIA, MAX_COPPIA, COPPIA_RANGE):
        if i < MAX_COPPIA - COPPIA_RANGE:
            COPPIA_INTERVALS.append((i, i + COPPIA_RANGE))
        else:
            COPPIA_INTERVALS.append((i, None))
    '''

    curr_pump_speed_index, curr_pump_speed = pump_speed_sig[t]
    if curr_pump_speed_index > 0:
        try:
            prev_index = [tup[0] for tup in pump_speed.values() if tup[0] < curr_pump_speed_index][-1]
            prev_pump_speed = pump_speed_sig.points[prev_index].value
        except IndexError:
            prev_pump_speed = None
    else:
        prev_pump_speed = curr_pump_speed

    curr_Tridu_index, curr_Tridu = Tridu_sig[t]
    if curr_Tridu_index > 0:
        try:
            prev_index = [tup[0] for tup in Tridu.values() if tup[0] < curr_Tridu_index][-1]
            prev_Tridu = Tridu_sig.points[prev_index].value
        except IndexError:
            prev_Tridu = None
    else:
        prev_Tridu = curr_Tridu

    curr_tmprt_index, curr_tmprt = tmprt_sig[t]
    if curr_tmprt_index > 0:
        try:
            prev_index = [tup[0] for tup in tmprt.values() if tup[0] < curr_tmprt_index][-1]
            prev_tmprt = tmprt_sig.points[prev_index].value
        except IndexError:
            prev_tmprt = None
    else:
        prev_tmprt = curr_tmprt

    '''
    if curr_coppia < MIN_COPPIA and (prev_coppia is not None and prev_coppia >= MIN_COPPIA):
        identified_event = events[-1]
    elif prev_coppia is None or abs(curr_coppia - prev_coppia) >= COPPIA_RANGE:
        for i, interval in enumerate(COPPIA_INTERVALS):
            if (i < len(COPPIA_INTERVALS) - 1 and interval[0] <= curr_coppia < interval[1]) or \
                    (i == len(COPPIA_INTERVALS) - 1 and curr_coppia >= interval[0]):
                identified_event = events[i]
    '''
    identified_event = None
    if prev_tmprt is not None: # for now we just ignore prev_tmprt None, but in case this function have to be revised
        # Identify event as in is_chg_pts
        for i, THRESHOLD in enumerate(range(MIN_TMPRT, MAX_TMPRT, TMPRT_RANGE)):
            if curr_tmprt < THRESHOLD <= prev_tmprt or prev_tmprt < THRESHOLD <= curr_tmprt:
                identified_event = events[i + (MAX_PUMP_SPEED - MIN_PUMP_SPEED)/PUMP_SPEED_RANGE + (MAX_TRIDU - MIN_TRIDU)/TRIDU_RANGE]

    if prev_Tridu is not None:  # for now we just ignore prev_tmprt None, but in case this function have to be revised
        for i, THRESHOLD in enumerate(range(MIN_TRIDU, MAX_TRIDU, TRIDU_RANGE)):
            if curr_Tridu < THRESHOLD <= prev_Tridu or prev_Tridu < THRESHOLD <= curr_Tridu:
                identified_event = events[i + (MAX_PUMP_SPEED - MIN_PUMP_SPEED)/PUMP_SPEED_RANGE]

    if prev_pump_speed is not None:  # for now we just ignore prev_tmprt None, but in case this function have to be revised
        for i, THRESHOLD in enumerate(range(MIN_PUMP_SPEED, MAX_PUMP_SPEED, PUMP_SPEED_RANGE)):
            if curr_pump_speed < THRESHOLD <= prev_pump_speed or prev_pump_speed < THRESHOLD <= curr_pump_speed:
                identified_event = events[i]  # I already know that there is an event for the pump speed

    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: datetime):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


def parse_data(path: str):
    pump_speed: SampledSignal = SampledSignal([], label='sp')
    Tridu: SampledSignal = SampledSignal([], label='Tr')
    tmprt: SampledSignal = SampledSignal([], label='tmp')

    dd_real = pd.read_csv(path)

    dd_pump_speed = dd_real[dd_real['DataObjectField'] == 'SpeedSP']
    dd_pump_speed.loc[:, 'TimeStamp'] = pd.to_datetime(dd_pump_speed['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_pump_speed.sort_values(by='TimeStamp')

    dd_Tridu = dd_real[dd_real['DataObjectField'] == 'TCuscinettiRiduttore']
    dd_Tridu.loc[:, 'TimeStamp'] = pd.to_datetime(dd_Tridu['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_Tridu.sort_values(by='TimeStamp')

    dd_Talim = dd_real[dd_real['DataObjectField'] == 'TCuscinettiAlimentazione']
    dd_Talim.loc[:, 'TimeStamp'] = pd.to_datetime(dd_Talim['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_Talim.sort_values(by='TimeStamp')

    dd_tmprt = dd_real[dd_real['DataObjectField'] == 'Value']
    dd_tmprt.loc[:, 'TimeStamp'] = pd.to_datetime(dd_tmprt['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_tmprt.sort_values(by='TimeStamp')

    #data_inizio_filtraggio = pd.to_datetime(DATA_INIZIO_FILTRO)
    #data_fine_filtraggio = pd.to_datetime(DATA_FINE_FILTRO)

    #dd_differenziale_dettaglio = dd_differenziale[(dd_differenziale['time'] >= data_inizio_filtraggio) & (dd_differenziale['time'] <= data_fine_filtraggio)]
    #dd_assorbimento_dettaglio = dd_assorbimento[(dd_assorbimento['time'] >= data_inizio_filtraggio) & (dd_assorbimento['time'] <= data_fine_filtraggio)]
    #dd_coppia_dettaglio = dd_coppia[(dd_coppia['time'] >= data_inizio_filtraggio) & (dd_coppia['time'] <= data_fine_filtraggio)]

    pump_speed.points.extend([SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_pump_speed.iterrows()])
    dd_Talim.points.extend([SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_Talim.iterrows()])
    dd_Tridu.points.extend([SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_Tridu.iterrows()])
    dd_tmprt.points.extend([SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_tmprt.iterrows()])

    return [dd_Talim, pump_speed, Tridu, tmprt]


def get_absorption_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_abs = sum([pt.value for pt in segment])
    avg_abs = sum_abs / (len(segment))
    return avg_abs


def plot_assorbimento_eventi(trace: TimedTrace):
    dd_real = pd.read_csv(
        'D:\\Uni\\Magistrale\\1 Anno\\1 semestre\\Software engineering 2\\Gr3n\\csv\\20250202_DecanterData_REAL.csv')
    dd_assorbimento = dd_real[dd_real['DataObjectField'] == 'Assorbimento']
    dd_assorbimento.loc[:, 'time'] = pd.to_datetime(dd_assorbimento['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_assorbimento = dd_assorbimento.sort_values(by='time')

    fig, ax = plt.subplots(figsize=(12, 6))

    xaxis_assorbimento = [record['time'] for index, record in dd_assorbimento.iterrows()]
    yaxis_assorbimento = [record['Value'] for index, record in dd_assorbimento.iterrows()]
    ax.plot(xaxis_assorbimento, yaxis_assorbimento, label='Assorbimento')

    for timestamp in trace.t:
        dt = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.min,
            timestamp.sec)
        ax.plot([dt], [dd_assorbimento['Value'].max()*1.10], 'rv')
        ax.vlines(x=dt, ymin=0, ymax=dd_assorbimento['Value'].max()*1.10, color='r', linestyle=':', alpha=0.5)

    # Formattazione dell'asse delle date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    DATA_INIZIO_FILTRO = config['GR3N']['DATA_INIZIO_FILTRO']
    DATA_FINE_FILTRO = config['GR3N']['DATA_FINE_FILTRO']
    ax.set_xlim(pd.to_datetime(DATA_INIZIO_FILTRO), pd.to_datetime(DATA_FINE_FILTRO))
    plt.xticks(rotation=45)

    plt.title('Assorbimento con Eventi')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

def plot_coppia_eventi(path: str, filename: str, trace: TimedTrace):
    dd_real = pd.read_csv(path)
    dd_coppia = dd_real[dd_real['DataObjectField'] == 'Coppia']
    dd_coppia.loc[:, 'time'] = pd.to_datetime(dd_coppia['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_coppia = dd_coppia.sort_values(by='time')

    fig, ax = plt.subplots(figsize=(12, 6))

    xaxis_assorbimento = [record['time'] for index, record in dd_coppia.iterrows()]
    yaxis_assorbimento = [record['Value'] for index, record in dd_coppia.iterrows()]
    ax.plot(xaxis_assorbimento, yaxis_assorbimento, label='Coppia')#, marker='o')

    for timestamp in trace.t:
        dt = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.min,
            timestamp.sec)
        ax.plot([dt], [dd_coppia['Value'].max()*1.10], 'rv')
        ax.vlines(x=dt, ymin=0, ymax=dd_coppia['Value'].max()*1.10, color='r', linestyle=':', alpha=0.5)

    # Formattazione dell'asse delle date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    plt.title(filename)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()