import configparser
import csv
import os
from typing import List, Tuple

from sha_learning.domain.lshafeatures import Event, FlowCondition
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from sha_learning.learning_setup.logger import Logger

config = configparser.ConfigParser()  # open the configuration file
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

try:
    CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))
except ValueError:
    CS_VERSION = None
SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])

LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    speed_cond = False
    if (curr[3] == 1 and prev[3] == 0) or (curr[3] == -1 and prev[3] == 0) or (curr[3] == 1 and prev[3] == -1):
        speed_cond = True
    return speed_cond or curr[1] != prev[1]


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    speed_sig = signals[1]
    pressure_sig = signals[2]
    speed = {pt.timestamp: (i, pt.value) for i, pt in enumerate(speed_sig.points)}
    pressure = {pt.timestamp: (i, pt.value) for i, pt in enumerate(pressure_sig.points)}

    # create a list of tuples where each tuple contains the limits of a speed interval
    SPEED_INTERVALS: List[Tuple[int, int]] = []
    for i in range(MIN_SPEED, MAX_SPEED, SPEED_RANGE):
        if i < MAX_SPEED - SPEED_RANGE:
            SPEED_INTERVALS.append((i, i + SPEED_RANGE))
        else:
            SPEED_INTERVALS.append((i, None))

    # identify the current and previous speed wrt the given timestamp t
    curr_speed_index, curr_speed = speed[t]
    if curr_speed_index > 0:
        try:
            prev_index = [tup[0] for tup in speed.values() if tup[0] < curr_speed_index][-1]
            prev_speed = speed_sig.points[prev_index].value
        except IndexError:
            prev_speed = None
    else:
        prev_speed = curr_speed

    # identify the current and previous pressure wrt the given timestamp t
    curr_press_index, curr_press = pressure[t]
    if curr_press_index > 0:
        try:
            prev_index = [tup[0] for tup in pressure.values() if tup[0] < curr_press_index][-1]
            prev_press = pressure_sig.points[prev_index].value
        except IndexError:
            prev_press = None
    else:
        prev_press = curr_press

    identified_event = None

    # if there is a pressure change, there is a load or unload event
    if curr_press != prev_press:
        # from 0 to 1 -> load
        if curr_press == 1.0 and prev_press == 0.0:
            identified_event = events[-2]
        # from 1 to 0 -> unload
        else:
            identified_event = events[-1]
    # if the previous velocity is bigger than the current one, we are going to 0 and we need to identify a stop event
    elif curr_speed < prev_speed:
        identified_event = events[-3]
    else:
        i = curr_speed_index
        while i < speed.__len__():
            const_speed = speed_sig.points[i].value
            if const_speed == speed_sig.points[i + 1].value and const_speed != 0:
                break
            else:
                i += 1
        # if the spindle is moving, return the constant speed that it will reach as a set point
        for i, interval in enumerate(SPEED_INTERVALS):
            if interval[0] <= const_speed <= interval[1]:
                identified_event = events[i]

    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(string: str):
    date = string.split('T')[0]
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    time = string.split('T')[1].split('Z')[0]
    hour = int(time[0:2])
    minute = int(time[3:5])
    second = int(time[6:8])
    return Timestamp(year, month, day, hour, minute, second)


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    power: SampledSignal = SampledSignal([], label='P')
    speed: SampledSignal = SampledSignal([], label='w')
    pressure: SampledSignal = SampledSignal([], label='pr')
    toolID: SampledSignal = SampledSignal([], label='id')
    speed_derivative: SampledSignal = SampledSignal([], label='wd')

    with open(path) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')

        for i, row in enumerate(reader):
            # to skip the columns name row
            if i == 0:
                continue

            # select the timestamp from column 0
            ts = parse_ts(row['_time'])

            # parse power value
            power.points.append(SignalPoint(ts, float(row['Total_power'])))

            # parse speed value: round to closest [100]
            speed_v = round(float(row['actual_Speed_SP1']) / 100) * 100
            speed.points.append(SignalPoint(ts, speed_v))

            # parse pallet pressure value
            pressure.points.append(SignalPoint(ts, float(row['Pressure'])))

            # parse tool ID value
            toolID.points.append(SignalPoint(ts, float(row['dictID'])))

            # parse a signal which represents the derivative of the speed vector
            if i > 0:
                if round(speed.points[i - 2].value) == round(speed.points[i - 1].value):  # if constant
                    speed_d = 0
                elif round(speed.points[i - 2].value) < round(speed.points[i - 1].value):  # if going up
                    speed_d = 1
                else:  # if going down
                    speed_d = -1
                speed_derivative.points.append(SignalPoint(ts, float(speed_d)))

        return [power, speed, pressure, toolID, speed_derivative]


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_power = sum([pt.value for pt in segment])
    avg_power = sum_power / len(segment)
    return avg_power
