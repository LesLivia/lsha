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

    @staticmethod
    def from_secs(secs: int):
        new_ts = Timestamp(int(secs / (365 * 24 * 3600)), 0, 0, 0, 0, 0)
        if secs > 365 * 24 * 3600:
            secs -= new_ts.year * 365 * 24 * 3600
        sum_d = 0
        for i, month_d in enumerate(DAYS_PER_MONTH):
            sum_d += month_d
            if secs <= sum_d * 24 * 3600:
                new_ts.month = i + 1
                secs -= (sum_d - month_d) * 24 * 3600
                break
        new_ts.day = int(secs / (24 * 3600)) + 1
        secs -= (new_ts.day - 1) * 24 * 3600
        new_ts.hour = int(secs / 3600)
        secs -= new_ts.hour * 3600
        new_ts.min = int(secs / 60)
        new_ts.sec = secs - new_ts.min * 60
        return new_ts

    def to_secs(self):
        months = sum(DAYS_PER_MONTH[:self.month - 1]) if self.month > 0 else 0
        days = self.year * 365 + months + self.day - 1 if self.day > 0 else 0
        minutes = self.hour * 60 + self.min
        seconds = minutes * 60 + self.sec
        return days * 24 * 3600 + seconds

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

    def __sub__(self, other):
        return self.to_secs() - other.to_secs()


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
    def __init__(self, guard, chan, symb):
        self.guard = guard
        self.chan = chan
        self.symbol = symb
        self.label = guard + ',' + chan if len(guard) > 0 else chan

    def __str__(self):
        return self.label

    def __eq__(self, other):
        return self.guard == other.guard and self.chan == other.chan

    def __hash__(self):
        return hash(self.label)


class ChangePoint:
    def __init__(self, t: Timestamp):
        self.t = t
        self.event = None

    def set_event(self, e: Event):
        self.event = e

    def __str__(self):
        return '{} -> {}'.format(self.t, self.event)

    def __eq__(self, other):
        return self.t == other.t and self.event == other.event
