from typing import Set

LOCATION_FORMATTER = 'q_{}'
FLOW_FORMATTER = 'f_{}'
DISTR_FORMATTER = 'D_{}'


class Location:
    def __init__(self, l_id: int, name: str, flow: int, distr: int, initial: bool = False, committed: bool = False):
        self.id = l_id
        self.name = name
        self.flow = flow
        self.distr = distr
        self.initial = initial
        self.committed = committed
        self.x = None
        self.y = None

    def set_res_id(self, res_id: str):
        self.res_id = res_id

    def __str__(self):
        return LOCATION_FORMATTER.format(self.id)

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def parse_loc(line: str):
        name = line.split(' ')[0].replace('\t', '')
        if name != '__init__':
            l_id = int(name.split('_')[1])
            label = line.split('<br/><b>')[1].replace('</b></FONT>>]\n', '')
            flow = int(label.split(', ')[0].replace('f_', ''))
            distr = int(label.split(', ')[1].replace('D_', ''))
        else:
            l_id = 999
            label = '__init__'
            flow = None
            distr = None

        return Location(l_id, name, flow, distr, name == '__init__')


class Edge:
    def __init__(self, start: Location, dest: Location, sync: str, guard: str = None, update: str = None):
        self.start = start
        self.dest = dest
        self.guard = guard
        self.sync = sync
        self.update = update

    def set_sensor_id(self, s_id: str):
        self.sensor_id = s_id

    def __str__(self):
        return '{} -> {}: {}, {}, {}'.format(str(self.start), str(self.dest), self.guard, self.sync, self.update)

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def parse_edge(line: str, locations: Set[Location]):
        start_name = line.split(' -> ')[0].replace('\t', '')
        dest_name = line.split(' -> ')[1].split(' [')[0]
        start = [l for l in locations if l.name == start_name][0]
        dest = [l for l in locations if l.name == dest_name][0]
        sync = line.split('COLOR="#0067b0">')[1].replace('</FONT>>]\n', '') + '!'

        return Edge(start, dest, sync)


class SHA:
    def __init__(self, name: str, locs: Set[Location], edges: Set[Edge]):
        self.name = name
        self.locations = locs
        self.edges = edges
