import configparser
import csv
from typing import List

from it.polimi.hri_learn.domain.lshafeatures import Event, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from it.polimi.hri_learn.lstar_sha.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'][0])
LOGGER = Logger('SUL DATA HANDLER')


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    pass


def parse_ts(ts: str):
    date = ts.split(' ')[0].split('-')
    time = ts.split(' ')[1].split(':')
    return Timestamp(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), int(time[2]))


def parse_data(path: str):
    # support method to parse traces sampled by ref query

    energy: SampledSignal = SampledSignal([], label='e')
    power: SampledSignal = SampledSignal([], label='P')
    speed: SampledSignal = SampledSignal([], label='w')
    pressure: SampledSignal = SampledSignal([], label='pr')

    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(reader):
            if i > 0:
                ts = parse_ts(row[1])

                # parse energy value: if no reading is avaiable,
                # retrieve last available measurement
                try:
                    energy_v = float(row[2].replace(',', '.'))
                except ValueError:
                    energy_v = None
                energy.points.append(SignalPoint(ts, energy_v))

                # parse speed value: round to closest [100]
                try:
                    speed_v = round(float(row[3].replace(',', '.')) / 100) * 100
                except ValueError:
                    speed_v = speed.points[-1].value
                speed.points.append(SignalPoint(ts, speed_v))

                # parse pallet pressure value
                try:
                    pressure_v = float(row[4].replace(',', '.'))
                except ValueError:
                    pressure_v = None
                pressure.points.append(SignalPoint(ts, pressure_v))

        # transform energy signal into power signal by computing
        # the difference betwen two consecutive measurements
        power_pts: List[SignalPoint] = [SignalPoint(energy.points[0].timestamp, 0.0)]
        last_reading = energy.points[0].value
        for i, pt in enumerate(energy.points):
            if i == 0:
                continue
            elif pt.value is None:
                power_pts.append(SignalPoint(pt.timestamp, None))
            else:
                power_pts.append(SignalPoint(pt.timestamp, 60 * (pt.value - last_reading)))
                last_reading = pt.value
        power.points = power_pts

        return [power, speed, pressure]


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    pass
