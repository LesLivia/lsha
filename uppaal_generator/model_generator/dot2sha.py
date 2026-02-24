from typing import Set

from uppaal_generator.model_generator.logger import Logger
from uppaal_generator.model_generator.sha import SHA, Location, Edge

LOGGER = Logger('Dot2Uppaal')


def parse_sha(path: str, name: str):
    LOGGER.info("Starting SHA .dot file parsing...")
    locs: Set[Location] = set()
    edges: Set[Edge] = set()

    with open(path, 'r') as dot_file:
        lines = dot_file.readlines()
        locs_lines = [l for l in lines if
                      (l.startswith('	q_') or l.startswith('	__init__')) and not l.__contains__('->')]
        for line in locs_lines:
            locs.add(Location.parse_loc(line))
        LOGGER.debug('Found {} locations.'.format(len(locs)))

        edge_lines = [l for l in lines if
                      (l.startswith('	q_') or l.startswith('	__init__')) and l.__contains__('->')]
        for line in edge_lines:
            edges.add(Edge.parse_edge(line, locs))
        LOGGER.debug('Found {} edges.'.format(len(edges)))

    new_sha = SHA(name, locs, edges)

    LOGGER.info("SHA correctly generated.")

    return new_sha
