import configparser
import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import skg_main.skg_mgrs.connector_mgr as conn
from skg_main.skg_mgrs.skg_reader import Skg_Reader

from uppaal_generator.model_generator.logger import Logger
from uppaal_generator.model_generator.sha import SHA, Edge, Location

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('semantic_main')[0] + 'semantic_main/resources/config/config.ini')
config.sections()

LOGGER = Logger('UppaalModelGenerator')

NTA_TPLT_PATH = config['MODEL GENERATION']['tplt.path']
NTA_TPLT_NAME = 'nta_template.xml'
MACHINE_TPLT_NAME = 'sha_template.xml'

INVARIANT_FUN = config['AUTOMATON']['invariant.merge']

LOCATION_TPLT = """<location id="{}" x="{}" y="{}">\n\t<name x="{}" y="{}">{}</name>
<label kind="invariant" x="{}" y="{}">{}</label>
</location>\n"""

N_RUNS = int(config['UPPAAL SETTINGS']['N_RUNS'])
TAU = int(config['UPPAAL SETTINGS']['TAU'])
QUERY_TPLT = """E[<={};{}](max: s.coll_Tcdf[{}])\n"""

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


def process_links(links: List[Link], edge: Edge, target: Location,
                  entity_to_int: Dict[str, int]):
    sync = edge.sync.replace('!', '')
    loc_entity = 1
    edge_entity = 1
    for link in links:
        link_edge = link.aut_feat[0].edge
        link_loc = link.aut_feat[0].loc
        link_entity = link.skg_feat[0].entity
        if link_edge is not None and link_edge.label == sync:
            edge_entity = entity_to_int[link_entity.entity_id]
        if link_loc is not None and target.name == link_loc.name:
            loc_entity = entity_to_int[link_entity.entity_id]

    return loc_entity, edge_entity


def get_dicts(links: List[Link]):
    ent_list = list(set([link.skg_feat[0].entity.entity_id for link in links]))
    return {x: i for i, x in enumerate(ent_list)}


def get_time_distr(name: str, start: int, end: int, loc_name: str):
    driver = conn.get_driver()
    reader: Skg_Reader = Skg_Reader(driver)

    formulae = reader.get_invariants(name, start, end, loc_name)
    x_mean = 0.0
    x_std = 0.0
    for i, f in enumerate(formulae):
        if INVARIANT_FUN.upper() == 'AVG':
            x_mean = (x_mean * i + f.params['mean']) / (i + 1)
            x_std = (x_std * i + f.params['std']) / (i + 1)

        if PLOT_DISTR:
            fig = plt.figure()
            plt.plot(f.params['cdfX'], f.params['cdfY'])
            plt.title(loc_name)
            plt.savefig(SAVE_PATH + loc_name + '.png')
            plt.close(fig)

    upp_th = x_mean + x_std
    low_th = x_mean - x_std

    driver.close()

    cdfX = [] if len(formulae) <= 0 else formulae[-1].params['cdfX']
    cdfY = [] if len(formulae) <= 0 else formulae[-1].params['cdfY']

    if config['AUTOMATON']['invariant.unit'] == 's':
        return low_th, upp_th, cdfX, cdfY
    else:
        return low_th / 100 / 60, upp_th / 100 / 60, cdfX, cdfY


def get_route_info(name: str, start: int, end: int, sync: str, loc_name: str):
    driver = conn.get_driver()
    reader: Skg_Reader = Skg_Reader(driver)

    route_info = reader.get_prob_weights(name, start, end, sync, loc_name)

    prob_weight = 0.0 if len(route_info) > 0 else 1.0
    for i, r in enumerate(route_info):
        prob_weight = (prob_weight * i + r[0]) / (i + 1)

    driver.close()

    return prob_weight


def sha_to_upp_tplt(learned_sha: SHA, name: str, start, end, links: List[Link]):
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
        time_distr = get_time_distr(name, start, end, loc.name)

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

        link_params = process_links(links, edge, edge.dest, get_dicts(links))
        link_params_start = process_links(links, edge, edge.start, get_dicts(links))

        if link_params[0] != link_params_start[0]:
            guard = "x &gt;= {:.2f}".format(get_time_distr(name, start, end, edge.start.name)[0])
            if INVARIANT_FUN.upper() != 'AVG':
                update = 'sample_ecdf({})'.format(loc_to_distr[edge.dest.id])
            else:
                update = ''
            update += ", update_entities({}, {}), x=0".format(link_params[0], link_params[1])
        else:
            guard = "true"
            update = "update_entities({}, {})".format(link_params[0], link_params[1])

        route_info = get_route_info(name, start, end, edge.sync.replace("!", ""), edge.start.name)

        if route_info >= 1.0:
            new_edge_str = EDGE_TPLT.format(start_id, dest_id,
                                            mid_x, mid_y, guard,
                                            mid_x, mid_y + 5, edge.sync,
                                            mid_x, mid_y + 10, update)
            edges_str += new_edge_str
        else:
            req_branch_point.append((edge, mid_x, mid_y, route_info, guard, update))

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

    entity_dict = ['{}: {}'.format(x, get_dicts(links)[x]) for x in get_dicts(links)]
    learned_sha_tplt = learned_sha_tplt.replace('**N_DICT**', str(len(entity_dict)))
    learned_sha_tplt = learned_sha_tplt.replace('**0.0_N_DICT**', ','.join(['0.0'] * len(entity_dict)))
    learned_sha_tplt = learned_sha_tplt.replace('**DICT**', '\n'.join(entity_dict))

    return learned_sha_tplt


def generate_query_file(name: str, links):
    query_path = SAVE_PATH + name + '.q'
    entity_dict = get_dicts(links)
    lines = set()
    for link in links:
        if link.aut_feat[0].loc is not None:
            lines.add(QUERY_TPLT.format(TAU, N_RUNS, entity_dict[link.skg_feat[0].entity.entity_id]))

    with open(query_path, 'w') as q_file:
        q_file.write(''.join(list(lines)))
    return query_path


def generate_upp_model(learned_sha: SHA, name: str, start, end, links: List[Link]):
    LOGGER.info("Starting Uppaal semantic_model generation...")

    # Learned SHA Management

    learned_sha_tplt = sha_to_upp_tplt(learned_sha, name, start, end, links)

    nta_path = (NTA_TPLT_PATH + NTA_TPLT_NAME).format(
        os.path.dirname(os.path.abspath(__file__)).split('semantic_main')[0] + 'semantic_main/')
    with open(nta_path, 'r') as nta_tplt:
        lines = nta_tplt.readlines()
        nta_tplt = ''.join(lines)

    unique_syncs = list(set([e.sync.replace('!', '') for e in learned_sha.edges]))
    nta_tplt = nta_tplt.replace('**CHANNELS**', ','.join(unique_syncs))
    nta_tplt = nta_tplt.replace('**MONITORS**', ','.join(['s.' + l.name for l in learned_sha.locations]))

    nta_tplt = nta_tplt.replace('**MACHINE**', learned_sha_tplt)
    nta_tplt = nta_tplt.replace('**TAU**', str(TAU))

    model_path = SAVE_PATH + name + '.xml'

    with open(model_path, 'w') as new_model:
        new_model.write(nta_tplt)

    LOGGER.info('Uppaal semantic_model successfully created.')

    query_path = generate_query_file(name, links)

    LOGGER.info('Uppaal query file successfully created.')

    return model_path, query_path
