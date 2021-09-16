from enum import Enum


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class POI:
    def __init__(self, name, coord):
        self.name = name
        self.coord = coord


class Pattern(Enum):
    FOLLOWER = 0
    LEADER = 1
    RECEIVER = 2


def getPattStr(value):
    if value == 0:
        return 'Follower'
    elif value == 1:
        return 'Leader'
    elif value == 2:
        return 'Recipient'


class Human:
    def __init__(self, name, upp_id, vel, pattern, p_f, p_fw):
        self.name = name
        self.upp_id = upp_id
        self.vel = vel
        self.pattern = pattern
        self.p_f = p_f
        self.p_fw = p_fw


class Experiment:
    def __init__(self, typ, time_bound):
        self.type = typ
        self.tb = time_bound


class Robot:
    def __init__(self, name, upp_id, vel, acc, bcharge, hToServe):
        self.name = name
        self.upp_id = upp_id
        self.vel = vel
        self.acc = acc
        self.bcharge = bcharge
        self.hToServe = hToServe
