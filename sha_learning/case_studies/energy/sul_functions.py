import configparser
import csv
import os
from typing import List, Tuple, Dict

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
PR_RANGE = int(config['ENERGY CS']['PR_RANGE'])
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])

LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    return abs(curr[0] - prev[0]) > SPEED_RANGE or abs(curr[1] - prev[1]) > PR_RANGE


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
    # if spindle was moving previously and now it is idle, return "stop" event
    if abs(curr_press - prev_press) > PR_RANGE:
        if curr_press > 500 and prev_press <= 500:
            identified_event = events[-2]
        else:
            identified_event = events[-1]
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
    date = ts.split(' ')[0].split('-')
    time = ts.split(' ')[1].split(':')
    return Timestamp(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), int(time[2]))


def parse_data(path: str):
    # support method to parse traces sampled by ref query

    energy: SampledSignal = SampledSignal([], label='e')
    power: SampledSignal = SampledSignal([], label='P')
    speed: SampledSignal = SampledSignal([], label='w')
    pressure: SampledSignal = SampledSignal([], label='pr')

    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        prev_p = 0

        for i, row in enumerate(reader):
            if i > 0:
                ts = parse_ts(row[1])

                # parse energy value: if no reading is avaiable,
                # retrieve last available measurement
                try:
                    energy_v = float(row[2].replace(',', '.'))
                except ValueError:
                    energy_v = None
                energy.points.append(SignalPoint(ts, energy_v))

                # parse speed value: round to closest [100]
                try:
                    speed_v = round(float(row[3].replace(',', '.')) / 100) * 100
                    speed_v = max(speed_v, 0)
                except ValueError:
                    if len(speed.points) > 0:
                        speed_v = speed.points[-1].value
                    else:
                        speed_v = 0.0
                speed.points.append(SignalPoint(ts, speed_v))

                # parse pallet pressure value
                try:
                    pressure_v = float(row[4].replace(',', '.'))
                    prev_p = pressure_v
                except ValueError:
                    pressure_v = prev_p
                pressure.points.append(SignalPoint(ts, pressure_v))

        # transform energy signal into power signal by computing
        # the difference betwen two consecutive measurements
        power_pts: List[SignalPoint] = [SignalPoint(energy.points[0].timestamp, 0.0)]
        last_reading = energy.points[0].value
        last_power_value = 0.0
        for i, pt in enumerate(energy.points):
            if i == 0:
                continue
            elif pt.value is None:
                power_pts.append(SignalPoint(pt.timestamp, last_power_value))
            else:
                if last_reading is not None:
                    power_pts.append(SignalPoint(pt.timestamp, 60 * (pt.value - last_reading)))
                    last_power_value = 60 * (pt.value - last_reading)
                else:
                    power_pts.append(SignalPoint(pt.timestamp, 60 * (0.0)))
                    last_power_value = 0.0
                last_reading = pt.value
        power.points = power_pts

        # filter speed signal
        filtered_speed_pts: List[SignalPoint] = []
        last_switch = Timestamp(speed.points[0].timestamp.year, speed.points[0].timestamp.month,
                                speed.points[0].timestamp.day, speed.points[0].timestamp.hour,
                                speed.points[0].timestamp.min, 0)
        max_batch = speed.points[0].value
        timestamps: List[Timestamp] = []
        value_to_mul: Dict[float, int] = dict()
        for i, pt in enumerate(speed.points):
            curr_ts = Timestamp(pt.timestamp.year, pt.timestamp.month, pt.timestamp.day,
                                pt.timestamp.hour, pt.timestamp.min, 0)
            if curr_ts == last_switch and i < len(speed.points) - 1:
                timestamps.append(pt.timestamp)
                max_batch = min(max_batch, pt.value)
                if pt.value in value_to_mul:
                    value_to_mul[pt.value] += 1
                else:
                    value_to_mul[pt.value] = 1
            elif curr_ts != last_switch or i == len(speed.points) - 1:
                max_mul = 0
                max_val = 0
                for val in value_to_mul:
                    if value_to_mul[val] > max_mul:
                        max_mul = value_to_mul[val]
                        max_val = val
                for t in timestamps:
                    filtered_speed_pts.append(SignalPoint(t, max_val))
                last_switch = curr_ts
                max_batch = pt.value
                value_to_mul.clear()
                timestamps = [curr_ts]

        filtered_speed = SampledSignal(filtered_speed_pts, label='w')

        # infer pressure signal if absent
        non_zero_pts = [pt for pt in pressure.points if pt.value > 0]
        if len(non_zero_pts) == 0:
            inferred_pressure: List[SignalPoint] = []
            last_non_zero_speed_ts = filtered_speed.points[0].timestamp
            for pt in filtered_speed.points:
                delta = pt.timestamp.to_secs() - last_non_zero_speed_ts.to_secs()
                if pt.value > MIN_SPEED or delta < 600:
                    if pt.value > MIN_SPEED:
                        last_non_zero_speed_ts = pt.timestamp
                    pressure_v = 800
                else:
                    pressure_v = 0
                inferred_pressure.append(SignalPoint(pt.timestamp, pressure_v))
            pressure = SampledSignal(inferred_pressure, label='pr')
        pressure.points[0].value = 0
        pressure.points[-2].value = 0
        pressure.points[-1].value = 0
        for pt in filtered_speed.points[:20]:
            pt.value = 0.0
        filtered_speed.points[-2].value = 0.0

        return [power, filtered_speed, pressure]


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_power = sum([pt.value for pt in segment])
    avg_power = sum_power / len(segment)
    return avg_power
