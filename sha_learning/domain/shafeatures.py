from enum import Enum
from typing import List


class LocLabels(Enum):
    IDLE = 'idle'
    BUSY = 'busy'


class Location:
    def __init__(self, name: str, flow_cond: str):
        self.name = name
        self.flow_cond = flow_cond

    def set_flow_cond(self, flow_cond: str):
        self.flow_cond = flow_cond

    def __eq__(self, other):
        same_name = self.name == other.name
        same_flow = self.flow_cond == other.flow_cond
        return same_name and same_flow

    def __str__(self):
        if self.flow_cond is not None:
            return self.name + ' ' + self.flow_cond
        else:
            return self.name

    def __hash__(self):
        return hash(str(self))


LOCATIONS: List[Location] = [Location(LocLabels.IDLE.value, None), Location(LocLabels.BUSY.value, None)]


class Edge:
    def __init__(self, start: Location, dest: Location, guard: str = '', sync: str = ''):
        self.start = start
        self.dest = dest
        self.guard = guard
        self.sync = sync

    def set_guard(self, guard):
        self.guard = guard

    def set_sync(self, sync):
        self.sync = sync

    def __eq__(self, other):
        same_start = self.start == other.start
        same_dest = self.dest == other.dest
        same_guard = self.guard == other.guard
        same_sync = self.sync == other.sync
        return same_start and same_dest and same_guard and same_sync


class StochasticHybridAutomaton:
    LOCATION_FORMATTER = 'q_{}'

    def __init__(self, loc: List[Location], edges: List[Edge]):
        self.locations = loc
        self.edges = edges

    def set_locations(self, loc: List[Location]):
        self.locations = loc

    def set_edges(self, edges: List[Edge]):
        self.edges = edges
