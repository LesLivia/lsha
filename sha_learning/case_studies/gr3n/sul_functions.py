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

COPPIA_MIDPOINT = int(config['GR3N']['COPPIA_MIDPOINT'])
LOGGER = Logger('SUL DATA HANDLER')
DATA_INIZIO_FILTRO = config['GR3N']['DATA_INIZIO_FILTRO']
DATA_FINE_FILTRO = config['GR3N']['DATA_FINE_FILTRO']
COPPIA_RANGE = int(config['GR3N']['COPPIA_RANGE'])
MIN_COPPIA = int(config['GR3N']['MIN_COPPIA'])
MAX_COPPIA = int(config['GR3N']['MAX_COPPIA'])

DIF_RANGE = int(config['GR3N']['DIF_RANGE'])
MIN_DIF = int(config['GR3N']['MIN_DIF'])
MAX_DIF = int(config['GR3N']['MAX_DIF'])


def is_chg_pt(curr, prev):
    return  (abs(curr[0] - prev[0]) > COPPIA_RANGE and (curr[0] < MAX_COPPIA or prev[0] < MAX_COPPIA) \
                    and (curr[0] > MIN_COPPIA or prev[0] > MIN_COPPIA)) or \
            (abs(curr[0] - prev[0]) > DIF_RANGE and (curr[0] < MAX_DIF or prev[0] < MAX_DIF) \
                    and (curr[0] > MIN_DIF or prev[0] > MIN_DIF))

def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    coppia_sig = signals[1]
    dif_sig = signals[2]
    coppia = {pt.timestamp: (i, pt.value) for i, pt in enumerate(coppia_sig.points)}
    diff = {pt.timestamp: (i, pt.value) for i, pt in enumerate(dif_sig.points)}

    COPPIA_INTERVALS: List[Tuple[int, int]] = []
    for i in range(MIN_COPPIA, MAX_COPPIA, COPPIA_RANGE):
        if i < MAX_COPPIA - COPPIA_RANGE:
            COPPIA_INTERVALS.append((i, i + COPPIA_RANGE))
        else:
            COPPIA_INTERVALS.append((i, None))

    DIF_INTERVALS: List[Tuple[int, int]] = []
    for i in range(MIN_DIF, MAX_DIF, DIF_RANGE):
        if i < MAX_DIF - DIF_RANGE:
            DIF_INTERVALS.append((i, i + DIF_RANGE))
        else:
            DIF_INTERVALS.append((i, None))

    curr_coppia_index, curr_coppia = coppia[t]
    if curr_coppia_index > 0:
        try:
            prev_index = [tup[0] for tup in coppia.values() if tup[0] < curr_coppia_index][-1]
            prev_coppia = coppia_sig.points[prev_index].value
        except IndexError:
            prev_coppia = None
    else:
        prev_coppia = curr_coppia

    curr_dif_index, curr_dif = diff[t]
    if curr_dif_index > 0:
        try:
            prev_index = [tup[0] for tup in diff.values() if tup[0] < curr_dif_index][-1]
            prev_dif = dif_sig.points[prev_index].value
        except IndexError:
            prev_dif = None
    else:
        prev_dif = curr_dif

    identified_event = None
    if curr_coppia < MIN_COPPIA and (prev_coppia is not None and prev_coppia >= MIN_COPPIA):
        identified_event = events[-1]
    elif prev_coppia is None or abs(curr_coppia - prev_coppia) >= COPPIA_RANGE:
        for i, interval in enumerate(COPPIA_INTERVALS):
            if (i < len(COPPIA_INTERVALS) - 1 and interval[0] <= curr_coppia < interval[1]) or \
                    (i == len(COPPIA_INTERVALS) - 1 and curr_coppia >= interval[0]):
                identified_event = events[i]

    if curr_dif < MIN_DIF and (prev_dif is not None and prev_dif >= MIN_DIF):
        identified_event = events[-1]
    elif prev_dif is None or abs(curr_dif - prev_dif) >= DIF_RANGE:
        for i, interval in enumerate(DIF_INTERVALS):
            if (i < len(DIF_INTERVALS) - 1 and interval[0] <= curr_dif < interval[1]) or \
                    (i == len(DIF_INTERVALS) - 1 and curr_dif >= interval[0]):
                identified_event = events[i + len(COPPIA_INTERVALS)]

    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: datetime):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


def parse_data(path: str):
    differenziale: SampledSignal = SampledSignal([], label='df')
    assorbimento: SampledSignal = SampledSignal([], label='a')
    coppia: SampledSignal = SampledSignal([], label='cp')

    dd_real = pd.read_csv(path)

    dd_differenziale = dd_real[dd_real['DataObjectField'] == 'Differenziale']
    dd_differenziale.loc[:, 'time'] = pd.to_datetime(dd_differenziale['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_differenziale.sort_values(by='time')

    dd_assorbimento = dd_real[dd_real['DataObjectField'] == 'Assorbimento']
    dd_assorbimento.loc[:, 'time'] = pd.to_datetime(dd_assorbimento['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_assorbimento.sort_values(by='time')

    dd_coppia = dd_real[dd_real['DataObjectField'] == 'Coppia']
    dd_coppia.loc[:, 'time'] = pd.to_datetime(dd_coppia['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_coppia.sort_values(by='time')

    #data_inizio_filtraggio = pd.to_datetime(DATA_INIZIO_FILTRO)
    #data_fine_filtraggio = pd.to_datetime(DATA_FINE_FILTRO)

    #dd_differenziale_dettaglio = dd_differenziale[(dd_differenziale['time'] >= data_inizio_filtraggio) & (dd_differenziale['time'] <= data_fine_filtraggio)]
    #dd_assorbimento_dettaglio = dd_assorbimento[(dd_assorbimento['time'] >= data_inizio_filtraggio) & (dd_assorbimento['time'] <= data_fine_filtraggio)]
    #dd_coppia_dettaglio = dd_coppia[(dd_coppia['time'] >= data_inizio_filtraggio) & (dd_coppia['time'] <= data_fine_filtraggio)]

    differenziale.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_differenziale.iterrows()])
    assorbimento.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_assorbimento.iterrows()])
    coppia.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_coppia.iterrows()])

    return [assorbimento, coppia, differenziale]


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