import configparser
import csv
import os
from typing import List, Tuple

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
SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])

LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    return abs(curr[0] - prev[0]) > SPEED_RANGE or curr[1] != prev[1]


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    speed_sig = signals[1]
    pressure_sig = signals[2]
    speed = {pt.timestamp: (i, pt.value) for i, pt in enumerate(speed_sig.points)}
    pressure = {pt.timestamp: (i, pt.value) for i, pt in enumerate(pressure_sig.points)}

    SPEED_INTERVALS: List[Tuple[int, int]] = []
    for i in range(MIN_SPEED, MAX_SPEED, SPEED_RANGE):
        if i < MAX_SPEED - SPEED_RANGE:
            SPEED_INTERVALS.append((i, i + SPEED_RANGE))
        else:
            SPEED_INTERVALS.append((i, None))

    curr_speed_index, curr_speed = speed[t]
    if curr_speed_index > 0:
        try:
            prev_index = [tup[0] for tup in speed.values() if tup[0] < curr_speed_index][-1]
            prev_speed = speed_sig.points[prev_index].value
        except IndexError:
            prev_speed = None
    else:
        prev_speed = curr_speed

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

    if curr_press != prev_press:
        if curr_press == 1.0 and prev_press == 0.0:
            identified_event = events[-2]
        else:
            identified_event = events[-1]
    # if spindle was moving previously and now it is idle, return "stop" event
    elif curr_speed < MIN_SPEED and (prev_speed is not None and prev_speed >= MIN_SPEED):
        identified_event = events[-3]
    else:
        # if spindle is now moving at a different speed than before,
        # return 'new speed' event, which varies depending on current speed range
        if prev_speed is None or abs(curr_speed - prev_speed) >= SPEED_RANGE:
            for i, interval in enumerate(SPEED_INTERVALS):
                if (i < len(SPEED_INTERVALS) - 1 and interval[0] <= curr_speed < interval[1]) or \
                        (i == len(SPEED_INTERVALS) - 1 and curr_speed >= interval[0]):
                    identified_event = events[i]

    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: str):
    fields = ts.split(':')
    return Timestamp(0, 0, 0, int(fields[0]), int(fields[1]), int(fields[2]))


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    power: SampledSignal = SampledSignal([], label='P')
    speed: SampledSignal = SampledSignal([], label='w')
    pressure: SampledSignal = SampledSignal([], label='pr')

    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        counter = 0

        for i, row in enumerate(reader):
            if i == 0:
                continue

            ts = parse_ts(row[2])

            if i > 1 and ts == speed.points[-1].timestamp:
                # parse power value
                power.points[-1].value = (power.points[-1].value * counter + float(row[4])) / (counter + 1)

                # parse speed value: round to closest [100]
                speed_v = round(float(row[3]) / 100) * 100
                speed.points[-1].value = min(speed_v, speed.points[-1].value)

                # parse pallet pressure value
                pressure_v = float(row[1] != 'UNLOAD')
                pressure.points[-1].value = min(pressure_v, pressure.points[-1].value)

                counter += 1
            else:
                counter = 0

                # parse power value
                power.points.append(SignalPoint(ts, float(row[4])))

                # parse speed value: round to closest [100]
                speed_v = round(float(row[3]) / 100) * 100
                speed.points.append(SignalPoint(ts, speed_v))

                # parse pallet pressure value
                pressure_v = float(not (row[1] == 'UNLOAD' or (row[1] == 'LOAD' and i == 1)))
                pressure.points.append(SignalPoint(ts, pressure_v))

        return [power, speed, pressure]


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_power = sum([pt.value for pt in segment])
    avg_power = sum_power / len(segment)
    return avg_power
