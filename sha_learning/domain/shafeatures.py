from enum import Enum
from typing import List, Dict, Set

from sha_learning.domain.lshafeatures import Trace
from sha_learning.learning_setup.logger import Logger


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

    def get_nondetermistic_edge(self, loc: Location):
        LOGGER = Logger('Non-Det. Check')

        outgoing_edges = [e for e in self.edges if e.start == loc]
        seen_events: Set[str] = set()
        for edge in outgoing_edges:
            if edge.sync in seen_events:
                LOGGER.warn('NON-DETERMINISM DETECTED! Location: {}, Event: {}'.format(loc.name, edge.sync))
                return edge.sync
            else:
                seen_events.add(edge.sync)
        return None

    def sanity_check(self, loc_dic: Dict[Trace, str]):
        LOGGER = Logger('Non-Det. Check')

        to_check: Set[Location] = set(self.locations)

        while len(to_check) > 0:
            for loc in self.locations:
                non_det_event = self.get_nondetermistic_edge(loc)
                # We only highlight non-deterministic edges purely to warn the user.
                # If the selected equality condition is 'strict', there should be no non-deterministic edges.
                # If the selected equality condition is 'weak', non-determinism is expected and---often---not solvable.
                # if non_det_event is not None:
                #    sha, merged = self.merge_loc(sha, loc, non_det_event, loc_dic)
                #    if merged:
                #        to_check = set(sha.locations)
                #        break
                #    else:
                #        LOGGER.warn('MERGING LOCATIONS UNSUCCESSFUL.')
                #        to_check = set()
                #        break
                to_check.remove(loc)
