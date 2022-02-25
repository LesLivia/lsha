import configparser
import sys
from typing import List, Dict

from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp, Event

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'][0])


def label_event(events: List[Event], signals: List[SampledSignal], symbols: Dict[str, str], t: Timestamp):
    wOpen = signals[2]
    heatOn = signals[0]
    t = t.to_secs()

    identified_guard = ''
    '''
    Repeat for every guard in the system
    '''
    curr_wOpen = list(filter(lambda x: x.timestamp.to_secs() <= t, wOpen.points))[-1]
    identified_guard += events[0].guard if curr_wOpen.value == 1.0 else '!' + events[0].guard
    if CS_VERSION == 'b' or CS_VERSION == 'c':
        identified_guard += events[1].guard if curr_wOpen.value == 2.0 else '!' + events[1].guard
        # identified_guard += self.get_guards()[2] if curr_wOpen.value == 0.0 else '!' + self.get_guards()[2]

    '''
    Repeat for every channel in the system
    '''
    curr_heatOn = list(filter(lambda x: x.timestamp.to_secs() <= t, heatOn.points))[-1]
    identified_channel = events[0].chan if curr_heatOn.value == 1.0 else events[1].chan

    '''
    Find symbol associated with guard-channel combination
    '''
    combination = identified_guard + ' and ' + identified_channel
    for key in symbols:
        if symbols[key] == combination:
            return key
