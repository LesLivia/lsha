import configparser
from typing import List

from it.polimi.hri_learn.domain.lshafeatures import Event, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from it.polimi.hri_learn.lstar_sha.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))

LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    return curr[1] != prev[1]


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    identified_event = None

    return identified_event


def parse_ts(ts: str):
    fields = ts.split(':')
    return Timestamp(0, 0, 0, int(fields[0]), int(fields[1]), int(fields[2]))


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    power: SampledSignal = SampledSignal([], label='P')
    speed: SampledSignal = SampledSignal([], label='w')
    pressure: SampledSignal = SampledSignal([], label='pr')

    return [power, speed, pressure]


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_power = sum([pt.value for pt in segment])
    avg_power = sum_power / len(segment)
    return avg_power
