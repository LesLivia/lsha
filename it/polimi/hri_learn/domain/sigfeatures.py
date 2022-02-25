from typing import List

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class Timestamp:
    def __init__(self, y: int, m: int, d: int, h: int, min: int, sec: float):
        self.year = y
        self.month = m
        self.day = d
        self.hour = h
        self.min = min
        self.sec = sec

    def to_secs(self):
        days = sum(DAYS_PER_MONTH[:self.month - 1]) + self.day - 1
        minutes = self.hour * 60 + self.min
        seconds = minutes * 60 + self.sec
        return days * 24 * 60 + seconds

    def __str__(self):
        return '{}/{}/{} {}:{}:{}'.format(self.day, self.month, self.year, self.hour, self.min, self.sec)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.to_secs() == other.to_secs()

    def __ge__(self, other):
        return self.to_secs() >= other.to_secs()

    def __lt__(self, other):
        return self.to_secs() < other.to_secs()


class SignalPoint:
    def __init__(self, t: Timestamp, val: float):
        self.timestamp = t
        self.value = val

    def __str__(self):
        return '{}: {}'.format(self.timestamp, self.value)

    def __eq__(self, other):
        return self.timestamp == other.timestamp and self.value == other.value


class SampledSignal:
    def __init__(self, pts: List[SignalPoint], label=None):
        self.label = label
        self.points = pts


class Event:
    def __init__(self, guard, chan):
        self.guard = guard
        self.chan = chan
        self.label = guard + '\n' + chan

    def __str__(self):
        return self.label

    def __eq__(self, other):
        return self.guard == other.guard and self.label == other.label


class ChangePoint:
    def __init__(self, t: Timestamp, evts: Event):
        self.t = t
        # we restrict to one event per changepoint
        self.event = evts

    def __str__(self):
        return '{} -> {}'.format(self.t, self.event)

    def __eq__(self, other):
        return self.t == other.t and self.event == other.event
