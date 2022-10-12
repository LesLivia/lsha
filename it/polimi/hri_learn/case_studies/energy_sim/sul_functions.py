import configparser
import csv
from typing import List, Tuple

from it.polimi.hri_learn.domain.lshafeatures import Event, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from it.polimi.hri_learn.lstar_sha.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))
SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])

LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    return abs(curr - prev) > SPEED_RANGE


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    speed_sig = signals[1]
    speed = {pt.timestamp: (i, pt.value) for i, pt in enumerate(speed_sig.points)}

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

    identified_event = None
    # if spindle was moving previously and now it is idle, return "stop" event
    if curr_speed < MIN_SPEED and (prev_speed is not None and prev_speed >= MIN_SPEED):
        identified_event = events[-1]
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
    return Timestamp(0, 0, 0, 0, int(ts), 0)


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    power: SampledSignal = SampledSignal([], label='P')
    speed: SampledSignal = SampledSignal([], label='w')
    pressure: SampledSignal = SampledSignal([], label='pr')

    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(reader):
            if i > 0:
                ts = parse_ts(row[1])

                # parse power value
                power.points.append(SignalPoint(ts, float(row[7])))

                # parse speed value: round to closest [100]
                speed_v = round(float(row[6]) / 100) * 100
                speed.points.append(SignalPoint(ts, speed_v))

                # parse pallet pressure value
                pressure_v = float(row[4].replace(',', '.'))
                pressure.points.append(SignalPoint(ts, pressure_v))

        return [power, speed, pressure]


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_power = sum([pt.value for pt in segment])
    avg_power = sum_power / len(segment)
    return avg_power
