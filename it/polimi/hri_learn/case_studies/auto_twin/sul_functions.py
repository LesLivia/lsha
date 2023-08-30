import configparser
from typing import List, Dict, Tuple

from it.polimi.hri_learn.domain.lshafeatures import Event, FlowCondition
from it.polimi.hri_learn.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from it.polimi.hri_learn.lstar_sha.logger import Logger
from src.ekg_extractor.mgrs.ekg_queries import SCHEMA_NAME

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))

LOGGER = Logger('SUL DATA HANDLER')

POV = config['AUTO-TWIN CONFIGURATION']['POV'].lower()


def is_chg_pt(curr, prev):
    return curr[0] != prev[0] and curr[0] > 0.0


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    curr_value = [pt.value for pt in signals[0].points if pt.timestamp == t][0]

    identified_event = [e for e in events if int(e.symbol.replace('s', '')) == int(curr_value)][0]

    return identified_event


def parse_ts(ts):
    try:
        return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.mins, ts.sec)
    except AttributeError:
        return Timestamp(0, 0, 0, 0, 0, ts)


def update_state_vector(path, state_vector: List[int], sensor_to_station: Dict[str, Tuple[int, str]]):
    if len(path) == 0:
        return
    else:
        last_sensor = path[0].activity.replace('Pass Sensor ', '')
        if sensor_to_station[last_sensor][1] is None:
            return update_state_vector(path[1:], state_vector, sensor_to_station)
        elif sensor_to_station[last_sensor][1] == 'IN':
            state_vector[sensor_to_station[last_sensor][0]] += 1
        else:
            state_vector[sensor_to_station[last_sensor][0]] -= 1
        return update_state_vector(path[1:], state_vector, sensor_to_station)


def bin_to_dec(bool_vector: List[int]):
    res = 0
    for i in range(0, len(bool_vector)):
        res += bool_vector[len(bool_vector) - 1 - i] * 2 ** i
    return res


def parse_value(path, i):
    # FIXME should be generic
    act_to_sensors = {"Entrada Material Sucio": 'S1', "Cargado en carro  L+D": 'S2',
                      "Carga L+D iniciada": 'S3', "Carga L+D liberada": 'S4',
                      "Montaje": 'S5', "Producción  montada": 'S6',
                      "Composición de cargas": 'S7', "Carga de esterilizador liberada": 'S8',
                      "Carga de esterilizadorliberada": 'S9'}
    if path[i].activity not in act_to_sensors:
        s_id = float(int(path[i].activity.replace('Pass Sensor ', '').replace('S', '')))
    else:
        sensor = act_to_sensors[path[i].activity]
        s_id = float(int(sensor.replace('S', '')))

    if POV == 'plant':
        # determine resource state vector
        # TODO this should become system-agnostic
        if SCHEMA_NAME == 'pizzaLineV1':
            state_vector = [0] * 5
            sensor_to_station = {'S1': (0, None), 'S2': (1, 'IN'), 'S3': (1, 'OUT'),
                                 'S7': (2, None), 'S4': (3, 'IN'), 'S5': (3, 'OUT'), 'S6': (4, None)}
        else:
            state_vector = [0] * 9
            sensor_to_station = {'S1': (0, None), 'S2': (1, 'IN'), 'S3': (1, 'OUT'),
                                 'S16': (2, None), 'S4': (3, 'IN'), 'S5': (3, 'OUT'),
                                 'S6': (4, 'IN'), 'S7': (4, 'OUT'), 'S8': (5, 'IN'),
                                 'S9': (5, 'OUT'), 'S10': (6, 'IN'), 'S13': (6, 'OUT'),
                                 'S12': (7, 'OUT'), 'S11': (7, 'IN'),
                                 'S14': (8, 'IN'), 'S15': (8, 'OUT')}

        update_state_vector(path[:i + 1], state_vector, sensor_to_station)
        idle_busy_vector = [int(v > 0) for v in state_vector]
        print(path[i].activity, state_vector, idle_busy_vector)

        return s_id, bin_to_dec(idle_busy_vector)
    else:
        return s_id


def parse_data(path):
    sensor_id: SampledSignal = SampledSignal([], label='s_id')
    sensor_id.points.append(SignalPoint(Timestamp(0, 0, 0, 0, 0, 0), 0))
    if POV == 'plant':
        state_signal: SampledSignal = SampledSignal([], label='state_vec')
        state_signal.points.append(SignalPoint(Timestamp(0, 0, 0, 0, 0, 0), 0))
    for i, ekg_event in enumerate(path):
        if ekg_event.date is None:
            ts = parse_ts(ekg_event.timestamp)
            if i < len(path) - 1:
                next_ts = parse_ts(path[i + 1].timestamp)
                new_tss = [Timestamp.from_secs(t) for t in range(ts.to_secs(), next_ts.to_secs(), 100)]
            else:
                new_tss = [ts]
        else:
            ts = parse_ts(ekg_event.date)
            if i < len(path) - 1:
                next_ts = parse_ts(path[i + 1].date)
                new_tss = [Timestamp.from_secs(t) for t in range(ts.to_secs(), next_ts.to_secs(), 100)]
            else:
                new_tss = [ts]

        # FIXME: this should be generic.
        if POV == 'plant':
            value, value_v = parse_value(path, i)
        else:
            value = parse_value(path, i)

        if len(new_tss) > 1:
            sensor_id.points.extend([SignalPoint(t, value) for t in new_tss[:-1]])
            sensor_id.points.append(SignalPoint(new_tss[-1], 0.0))
            if POV == 'plant':
                state_signal.points.extend([SignalPoint(t, value_v) for t in new_tss])
        elif len(new_tss) > 0:
            sensor_id.points.append(SignalPoint(new_tss[-1], value))
            if POV == 'plant':
                state_signal.points.append(SignalPoint(new_tss[-1], value_v))
        else:
            # in case there are two events at the same time, the last one overrides.
            sensor_id.points[-1].value = value
            if POV == 'plant':
                state_signal.points[-1].value = value_v

    last_ts = sensor_id.points[-1].timestamp
    sensor_id.points.append(
        SignalPoint(Timestamp(last_ts.year, last_ts.month, last_ts.day, last_ts.hour, last_ts.min, last_ts.sec + 1),
                    sensor_id.points[-1].value))

    if POV == 'plant':
        state_signal.points.append(
            SignalPoint(Timestamp(last_ts.year, last_ts.month, last_ts.day, last_ts.hour, last_ts.min, last_ts.sec + 1),
                        state_signal.points[-1].value))

        return [sensor_id, state_signal]
    else:
        return [sensor_id]


def get_rand_param(segment: List[SignalPoint], flow: FlowCondition):
    return segment[0].value
