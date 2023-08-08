import configparser
from typing import List

from it.polimi.hri_learn.domain.lshafeatures import Event, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from it.polimi.hri_learn.lstar_sha.logger import Logger
import src.ekg_extractor.model.schema as ekg_schema

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))

LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    return curr[0] != prev[0]


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    curr_value = [pt.value for pt in signals[0].points if pt.timestamp == t][0]

    identified_event = [e for e in events if int(e.symbol.replace('s', '')) == int(curr_value)][0]

    return identified_event


def parse_ts(ts: ekg_schema.Timestamp):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.mins, ts.sec)


def parse_data(path):
    sensor_id: SampledSignal = SampledSignal([], label='s_id')
    sensor_id.points.append(SignalPoint(Timestamp(0, 0, 0, 0, 0, 0), 0))
    for ekg_event in path:
        sensor_id.points.append(SignalPoint(parse_ts(ekg_event.date), float(int(ekg_event.activity.replace('S', '')))))

    last_ts = sensor_id.points[-1].timestamp
    sensor_id.points.append(
        SignalPoint(Timestamp(last_ts.year, last_ts.month, last_ts.day, last_ts.hour, last_ts.min, last_ts.sec + 1),
                    sensor_id.points[-1].value))

    return [sensor_id]


def get_rand_param(segment: List[SignalPoint], flow: FlowCondition):
    return segment[0].value
