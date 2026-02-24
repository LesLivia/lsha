import configparser
import os
from typing import List, Dict, Tuple

import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer

from sha_learning.case_studies.lego_factory.sul_functions import ACT_TO_SENSORS
from sha_learning.domain.sigfeatures import Timestamp
from uppaal_generator.model_generator.logger import Logger
from uppaal_generator.model_generator.sha import SHA, Edge

config = configparser.ConfigParser()
config.sections()
config.read('uppaal_generator/resources/config.ini')
config.sections()

LOGGER = Logger('UppaalModelGenerator')

NTA_TPLT_PATH = config['MODEL GENERATION']['tplt.path']
NTA_TPLT_NAME = 'nta_template.xml'
MACHINE_TPLT_NAME = 'sha_template.xml'

INVARIANT_FUN = config['AUTOMATON']['invariant.merge']

LOCATION_TPLT = """<location id="{}" x="{}" y="{}">\n\t<name x="{}" y="{}">{}</name>
<label kind="invariant" x="{}" y="{}">{}</label>
</location>\n"""

X_START = 0
X_MAX = 900
X_RANGE = 300
Y_START = 0
Y_RANGE = 300

EDGE_TPLT = """\n<transition>\n\t<source ref="{}"/>\n\t<target ref="{}"/>
\t<label kind="guard" x="{}" y="{}">{}</label>
\t<label kind="synchronisation" x="{}" y="{}">{}</label>
\t<label kind="assignment" x="{}" y="{}">{}</label>
</transition>"""

SAVE_PATH = config['MODEL GENERATION']['save.path'].format(
    os.path.dirname(os.path.abspath(__file__)).split('semantic_main')[0] + 'semantic_main/')

PROB_EDGE_TPLT = """<transition>\n\t<source ref="{}"/>\n\t<target ref="{}"/>
\t<label kind="guard" x="{}" y="{}">{}</label>
\t<label kind="synchronisation" x="{}" y="{}">{}</label>
\t<label kind="assignment" x="{}" y="{}">{}</label>
\t<label kind="probability" x="{}" y="{}">{}</label></transition>"""

BRANCH_POINT_TPLT = """<branchpoint id="{}" x="{}" y="{}"/>\n"""

BRANCH_POINT_EDGE_TPLT = """"<transition>\n\t<source ref="{}"/>\n\t<target ref="{}"/>\n</transition>"""

TIME_DISTR = """const double ECDFx_{}[{}] = {};
const double ECDFy_{}[{}] = {};\n
"""

ECDF_SAMPLING_TPLT = """
        for(i=0; i &lt; ECDF_SIZES[d]-1 &amp;&amp; not found;i++)
			if(ECDFy_{}[i] &gt; pr) {{
                Tcdf = ECDFx_{}[i];
                found = true;
            }}
        if(not found) Tcdf = ECDFx_{}[i];
"""

FUNC_TPLT = "{} if (d == {}) {{ {} }}"

PLOT_DISTR = config['AUTOMATON']['plot.cdf'].lower() == 'true'


def extract_event_station_associations():
    xes_path = config['MODEL GENERATION']['xes.path']
    log = xes_importer.apply(xes_path)

    associations = dict()

    for trace in log:
        for i, event in enumerate(trace):
            station_id = event["station_id"]
            event_id = event["org:resource"]
            if station_id not in associations:
                associations[station_id] = {"s" + str(ACT_TO_SENSORS.index(event["org:resource"]) + 1)}
            else:
                associations[station_id].update({"s" + str(ACT_TO_SENSORS.index(event["org:resource"]) + 1)})

    return associations


def parse_ts(ts):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond / 1000)


def extract_time_distributions(start_date, end_date):
    """
    Extract processing time distribution for a given station_id (loc_name)
    within a global time window [start, end].

    Returns:
        low_th, upp_th, cdfX, cdfY
    """
    xes_path = config['MODEL GENERATION']['xes.path']
    log = xes_importer.apply(xes_path)

    durations = dict()

    for trace in log:
        for i, event in enumerate(trace):
            ts = parse_ts(event["time:timestamp"])

            # Only start segment if event is in time window
            if ts < start_date:
                break

            if i == 0 or event["station_id"] != trace[i - 1]["station_id"]:
                segment_start_time = ts

            # this is the last event or
            if (i == len(trace) - 1 or
                    # or the next event is in a different station
                    (i < len(event) - 1 and (event["station_id"] != trace[i + 1]["station_id"]
                                             # or the next event is in the same station but the next point is after the end date
                                             or parse_ts(trace[i + 1]["time:timestamp"]) > end_date))):
                segment_end_time = parse_ts(trace[i + 1]["time:timestamp"])
                duration = (segment_end_time.to_millis() - segment_start_time.to_millis()) / 1000.0
                if event["station_id"] in durations:
                    durations[event["station_id"]].append(duration)
                else:
                    durations[event["station_id"]] = [duration]

            if parse_ts(trace[i + 1]["time:timestamp"]) > end_date:
                break

    distributions = dict()

    # Statistics
    for station in durations:
        values = durations[station]
        x_mean = np.mean(values)
        x_std = np.std(values)

        low_th = x_mean - x_std
        upp_th = x_mean + x_std

        # Empirical CDF
        sorted_durations = np.sort(values)
        cdfX = sorted_durations.tolist()
        cdfY = (np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)).tolist()

        distributions[station] = (low_th, upp_th, cdfX, cdfY)

    return distributions


def extract_prob_weights(start_date, end_date):
    """
    Extract processing time distribution for a given station_id (loc_name)
    within a global time window [start, end].

    Returns:
        low_th, upp_th, cdfX, cdfY
    """
    xes_path = config['MODEL GENERATION']['xes.path']
    log = xes_importer.apply(xes_path)

    occurrences = dict()

    for trace in log:
        for i, event in enumerate(trace):
            ts = parse_ts(event["time:timestamp"])

            # Only start segment if event is in time window
            if ts < start_date or ts > end_date or i == len(trace) - 1:
                break

            current_station = event["station_id"]
            next_station = trace[i + 1]["station_id"]

            if current_station not in occurrences:
                occurrences[current_station] = [next_station]
            else:
                occurrences[current_station].append(next_station)

    weights = dict()

    # Statistics
    for station in occurrences:
        for occ in occurrences[station]:
            if (station, occ) not in weights:
                weights[(station, occ)] = sum([x == occ for x in occurrences[station]]) / len(occurrences[station])

    return weights


def link_locations_w_params(learned_sha, distributions,
                            event_station_associations):
    locations_to_distributions = dict()
    locations_to_distributions["__init__"] = (0.0, 0.0, [], [])

    for edge in learned_sha.edges:
        for station in event_station_associations:
            if edge.sync.replace("!", "") in event_station_associations[station]:
                locations_to_distributions[edge.dest.name] = distributions[station]

    return locations_to_distributions


def locations_to_stations(learned_sha, event_station_associations):
    locations_to_stations = dict()
    locations_to_stations["__init__"] = "Start"
    for edge in learned_sha.edges:
        for station in event_station_associations:
            if edge.sync.replace("!", "") in event_station_associations[station]:
                locations_to_stations[edge.dest.name] = station

    return locations_to_stations


def get_route_info(name: str, start: int, end: int, sync: str, loc_name: str):
    # driver = conn.get_driver()
    # reader: Skg_Reader = Skg_Reader(driver)
    #
    # route_info = reader.get_prob_weights(name, start, end, sync, loc_name)
    #
    # prob_weight = 0.0 if len(route_info) > 0 else 1.0
    # for i, r in enumerate(route_info):
    #     prob_weight = (prob_weight * i + r[0]) / (i + 1)
    #
    # driver.close()
    #
    # return prob_weight
    return None


def sha_to_upp_tplt(learned_sha: SHA, name: str, start, end,
                    loc_to_stations,
                    loc_to_distributions,
                    probability_weights):
    machine_path = (NTA_TPLT_PATH + MACHINE_TPLT_NAME).format(
        os.path.dirname(os.path.abspath(__file__)).split('semantic_main')[0] + 'semantic_main/')
    with open(machine_path, 'r') as machine_tplt:
        lines = machine_tplt.readlines()
        learned_sha_tplt = ''.join(lines)

    locations_str = ''
    x = X_START
    y = Y_START

    cdf_str = ''
    sizes_str = 'const int ECDF_SIZES[{}] = {{'.format(len(learned_sha.locations))
    func_str = ''

    loc_to_distr = {}

    for i, loc in enumerate(learned_sha.locations):
        time_distr = loc_to_distributions[loc.name]

        if INVARIANT_FUN.upper() == 'AVG':
            invariant = "x &lt;= {:.2f}".format(time_distr[1])
        else:
            sizes_str += str(len(time_distr[2]))
            if i != len(learned_sha.locations) - 1:
                sizes_str += ','

            if len(time_distr[2]) <= 0:
                invariant = "x &lt;= {:.2f}".format(time_distr[1])
            else:
                loc_to_distr[loc.id] = i

                invariant = "x &lt;= Tcdf"
                if config['AUTOMATON']['invariant.unit'] == 's':
                    x_vals = '{' + ','.join(['{:.1f}'.format(x) for x in time_distr[2]]) + '}'
                else:
                    x_vals = '{' + ','.join(['{:.1f}'.format(x / 100 / 60) for x in time_distr[2]]) + '}'
                y_vals = '{' + ','.join(['{:.4f}'.format(x) for x in time_distr[3]]) + '}'
                cdf_str += TIME_DISTR.format(i, len(time_distr[2]), x_vals,
                                             i, len(time_distr[3]), y_vals)

                if i == 0:
                    func_str += FUNC_TPLT.format('', i, ECDF_SAMPLING_TPLT.format(i, i, i))
                else:
                    func_str += FUNC_TPLT.format('else', i, ECDF_SAMPLING_TPLT.format(i, i, i))

        new_loc_str = LOCATION_TPLT.format('id' + str(loc.id), x, y, x, y - 20, loc.name,
                                           x, y - 30, invariant)

        loc.x = x
        loc.y = y
        locations_str += new_loc_str

        if loc.initial:
            learned_sha_tplt = learned_sha_tplt.replace('**INIT_ID**', 'id' + str(loc.id))

        if x < X_MAX:
            x = x + X_RANGE
        else:
            x = X_START
            y = y + Y_RANGE

    req_branch_point: List[Tuple[Edge, float, float, float, str, str]] = []
    edges_str = ''
    for edge in learned_sha.edges:
        start_id = 'id' + str(edge.start.id)
        dest_id = 'id' + str(edge.dest.id)
        x1, y1, x2, y2 = edge.start.x, edge.start.y, edge.dest.x, edge.dest.y
        mid_x = abs(x1 - x2) / 2 + min(x1, x2)
        mid_y = abs(y1 - y2) / 2 + min(y1, y2)

        station_start = loc_to_stations[edge.start.name]
        station_dest = loc_to_stations[edge.dest.name]

        if station_start != station_dest:
            time_distr_start = loc_to_distributions[edge.start.name]
            guard = "x &gt;= {:.2f}".format(time_distr_start[0])
            if INVARIANT_FUN.upper() != 'AVG':
                update = 'sample_ecdf({})'.format(loc_to_distr[edge.dest.id])
            else:
                update = ''
            update += ", x=0"
        else:
            guard = "true"
            update = ''

        if station_start == "Start":
            prob_weight = 1.0
        else:
            prob_weight = probability_weights[(station_start, station_dest)]

        if prob_weight >= 1.0:
            new_edge_str = EDGE_TPLT.format(start_id, dest_id,
                                            mid_x, mid_y, guard,
                                            mid_x, mid_y + 5, edge.sync,
                                            mid_x, mid_y + 10, update)
            edges_str += new_edge_str
        else:
            req_branch_point.append((edge, mid_x, mid_y, prob_weight, guard, update))

    conn_sets: Dict[str, int] = {}
    bp_id = 1000
    for tup in req_branch_point:
        if not tup[0].start.name in conn_sets:
            bp_str = BRANCH_POINT_TPLT.format("id{}".format(bp_id), tup[1], tup[2])
            locations_str += bp_str
            bp_edge_str = BRANCH_POINT_EDGE_TPLT.format("id{}".format(tup[0].start.id), "id{}".format(bp_id))
            edges_str += bp_edge_str
            conn_sets[tup[0].start.name] = bp_id
            bp_id += 1
        edge_str = PROB_EDGE_TPLT.format("id{}".format(conn_sets[tup[0].start.name]),
                                         "id{}".format(tup[0].dest.id),
                                         tup[1], tup[2], tup[4],
                                         tup[1], tup[2] + 5, tup[0].sync,
                                         tup[1], tup[2] + 10, tup[5],
                                         tup[1], tup[2] + 15, tup[3])
        edges_str += edge_str

    learned_sha_tplt = learned_sha_tplt.replace('**LOCATIONS**', locations_str)
    learned_sha_tplt = learned_sha_tplt.replace('**TRANSITIONS**', edges_str)
    learned_sha_tplt = learned_sha_tplt.replace('**TCDF**', sizes_str + '};\n\n' + cdf_str)
    learned_sha_tplt = learned_sha_tplt.replace('**SAMPLING_FN**', func_str)

    return learned_sha_tplt


def generate_upp_model(learned_sha: SHA, name: str, start, end):
    LOGGER.info("Starting Uppaal model generation...")

    event_station_associations = extract_event_station_associations()
    loc_to_station_dict = locations_to_stations(learned_sha, event_station_associations)
    distributions = extract_time_distributions(start, end)
    probability_weights = extract_prob_weights(start, end)
    locations_to_distributions = link_locations_w_params(learned_sha, distributions,
                                                         event_station_associations)

    # Learned SHA Management

    learned_sha_tplt = sha_to_upp_tplt(learned_sha, name, start, end,
                                       loc_to_station_dict,
                                       locations_to_distributions,
                                       probability_weights)

    nta_path = (NTA_TPLT_PATH + NTA_TPLT_NAME).format(
        os.path.dirname(os.path.abspath(__file__)).split('semantic_main')[0] + 'semantic_main/')
    with open(nta_path, 'r') as nta_tplt:
        lines = nta_tplt.readlines()
        nta_tplt = ''.join(lines)

    unique_syncs = list(set([e.sync.replace('!', '') for e in learned_sha.edges]))
    nta_tplt = nta_tplt.replace('**CHANNELS**', ','.join(unique_syncs))
    nta_tplt = nta_tplt.replace('**MONITORS**', ','.join(['s.' + l.name for l in learned_sha.locations]))

    nta_tplt = nta_tplt.replace('**MACHINE**', learned_sha_tplt)
    nta_tplt = nta_tplt.replace('**TAU**', "100")

    model_path = SAVE_PATH + name + '.xml'

    with open(model_path, 'w') as new_model:
        new_model.write(nta_tplt)

    LOGGER.info('Uppaal semantic_model successfully created.')

    return model_path
