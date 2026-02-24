import configparser
import os
from typing import List

from sha_learning.domain.lshafeatures import FlowCondition
from sha_learning.domain.sigfeatures import SampledSignal, SignalPoint, Timestamp, Event

config = configparser.ConfigParser()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS = config['SUL CONFIGURATION']['CASE_STUDY']

ACT_TO_SENSORS = ['corner1_START', 'corner1_RETURN', 'corner1_TRANSFER',
                  'corner2_START', 'corner2_RETURN', 'corner2_TRANSFER',
                  'End',
                  'splitter1_FORWARD', 'splitter1_RETURN', 'splitter1_TRANSFER',
                  'splitter2_FORWARD', 'splitter2_RETURN', 'splitter2_TRANSFER',
                  'splitter3_FORWARD', 'splitter3_RETURN', 'splitter3_TRANSFER',
                  'splitter4_FORWARD', 'splitter4_RETURN', 'splitter4_TRANSFER',
                  'splitter5_CHECKOUT', 'splitter5_FINISH', 'splitter5_FORWARD', 'splitter5_RETURN',
                  'splitter5_SCRAP', 'splitter5_TRANSFER',
                  'station11_FAIL', 'station11_LOAD', 'station11_PROCESS', 'station11_TRANSFER',
                  'station11_UNLOAD',
                  'station21_FAIL', 'station21_LOAD', 'station21_PASS', 'station21_pass_PASS', 'station21_PROCESS',
                  'station21_TRANSFER', 'station21_pass_TRANSFER', 'station21_UNLOAD',
                  'station22_FAIL', 'station22_LOAD', 'station22_PASS', 'station22_pass_PASS', 'station22_PROCESS',
                  'station22_TRANSFER', 'station22_pass_TRANSFER', 'station22_UNLOAD', 'station31_BLOCK',
                  'station31_FAIL', 'station31_LOAD', 'station31_PASS', 'station31_pass_PASS', 'station31_PROCESS',
                  'station31_TRANSFER', 'station31_pass_TRANSFER', 'station31_UNLOAD',
                  'station41_FAIL', 'station41_LOAD', 'station41_PASS', 'station41_pass_PASS', 'station41_PROCESS',
                  'station41_TRANSFER', 'station41_pass_TRANSFER', 'station41_UNLOAD',
                  'station51_FAIL', 'station51_LOAD', 'station51_PASS', 'station51_pass_PASS', 'station51_PROCESS',
                  'station51_TRANSFER', 'station51_pass_TRANSFER', 'station51_UNLOAD',
                  'station52_FAIL', 'station52_LOAD', 'station52_PASS', 'station52_pass_PASS', 'station52_PROCESS',
                  'station52_TRANSFER', 'station52_pass_TRANSFER', 'station52_UNLOAD',
                  'station61_LOAD', 'station61_PASS', 'station61_pass_PASS', 'station61_PROCESS',
                  'station61_TRANSFER', 'station61_pass_TRANSFER', 'station61_UNLOAD',
                  'station71_LOAD', 'station71_PASS', 'station71_pass_PASS', 'station71_PROCESS',
                  'station71_TRANSFER', 'station71_pass_TRANSFER', 'station71_UNLOAD']


def is_chg_pt(curr, prev):
    return curr[0] != prev[0] and curr[0] > 0.0


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    curr_value = [pt.value for pt in signals[0].points if pt.timestamp == t][0]

    identified_event = [e for e in events if int(e.symbol.replace('s', '')) == int(curr_value)][0]

    return identified_event


def parse_ts(ts):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond / 1000)


def get_acquisition_bounds():
    start_fields = config["AUTO-TWIN CONFIGURATION"]["START_DATE"].split('-')
    end_fields = config["AUTO-TWIN CONFIGURATION"]["END_DATE"].split('-')

    start_date = Timestamp(int(start_fields[0]), int(start_fields[1]), int(start_fields[2]), int(start_fields[3]),
                           int(start_fields[4]), int(start_fields[5]))
    end_date = Timestamp(int(end_fields[0]), int(end_fields[1]), int(end_fields[2]), int(end_fields[3]),
                         int(end_fields[4]), int(end_fields[5]))

    return start_date, end_date


def parse_data(path):
    DELTA_T = 300

    start_date, end_date = get_acquisition_bounds()

    sensor_id: SampledSignal = SampledSignal([], label='s_id')
    sensor_id.points.append(SignalPoint(Timestamp(0, 0, 0, 0, 0, 0), 0))
    for i, event in enumerate(path):
        ts = parse_ts(event["time:timestamp"])

        if ts < start_date or ts > end_date:
            continue

        if i < len(path) - 1:
            next_ts = parse_ts(path[i + 1]["time:timestamp"])
            new_tss = [Timestamp.from_millis(t) for t in range(ts.to_millis(), next_ts.to_millis(), DELTA_T)]
        else:
            new_tss = [ts]

        value = ACT_TO_SENSORS.index(event["org:resource"]) + 1

        if i > 0 and ts == parse_ts(path[i - 1]["time:timestamp"]):
            # in case there are two events at the same time, the last one overrides.
            sensor_id.points[-1].value = value
        elif len(new_tss) > 1:
            sensor_id.points.extend([SignalPoint(t, value) for t in new_tss[:-1]])
            sensor_id.points.append(SignalPoint(new_tss[-1], 0.0))
        elif len(new_tss) > 0:
            sensor_id.points.append(SignalPoint(new_tss[-1], value))
            if i < len(path) - 1:
                next_ts = parse_ts(path[i + 1]["time:timestamp"])
                sensor_id.points.append(SignalPoint(Timestamp.from_secs(next_ts.to_secs() - 1), 0.0))
        else:
            sensor_id.points.append(SignalPoint(ts, value))

    last_ts = sensor_id.points[-1].timestamp
    sensor_id.points.append(
        SignalPoint(Timestamp(last_ts.year, last_ts.month, last_ts.day, last_ts.hour, last_ts.min, last_ts.sec + 1),
                    sensor_id.points[-1].value))

    return [sensor_id]


def get_rand_param(segment: List[SignalPoint], flow: FlowCondition):
    return segment[0].value
