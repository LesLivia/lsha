from enum import Enum
from typing import List


class Position:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

    @staticmethod
    def parse_pos(s: str):
        fields = s.split('#')
        return Position(float(fields[0]), float(fields[1]))


class SignalPoint:
    def __init__(self, t: float, humId: int, val):
        self.timestamp = t
        self.humId = humId
        self.value = val
        self.notes: List[str] = []

    def __str__(self):
        return '(hum {}) {}: {}'.format(self.humId, self.timestamp, self.value)


class SignalType(Enum):
    NUMERIC = 'float'
    POSITION = 'pos'


class TimeInterval:
    def __init__(self, t_min: float, t_max: float):
        self.t_min = t_min
        self.t_max = t_max


class Labels(Enum):
    STARTED = 'walk'
    STOPPED = 'stop'


class ChangePoint:
    def __init__(self, t: TimeInterval, label: Labels):
        self.dt = t
        self.event = label

    def __str__(self):
        return '({}, {}) -> {}'.format(self.dt.t_min, self.dt.t_max, self.event)


class Event:
    def __init__(self, timestamp, label):
        self.timestamp = timestamp
        self.label = label

    def __str__(self):
        return '{:.2f}: {}'.format(self.timestamp, self.label)
