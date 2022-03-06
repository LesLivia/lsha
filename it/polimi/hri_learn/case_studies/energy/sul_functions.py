from typing import List

from it.polimi.hri_learn.domain.lshafeatures import Event, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    pass


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    pass


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    pass
