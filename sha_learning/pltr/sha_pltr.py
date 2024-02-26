from typing import List

from graphviz import Digraph

from sha_learning.domain.shafeatures import StochasticHybridAutomaton

FONT_OPEN_TAG = '<FONT {}>'
FONT_CLOSE_TAG = '</FONT>'
FONT_COLOR_ATTR = ' COLOR="{}"'
FONT_SIZE_ATTR = ' POINT-SIZE="{}"'


def style_label(args: List[str], font_sizes: List[int] = None, font_colors: List[str] = None):
    label = '<'
    for (index, arg) in enumerate(args):
        font_attrs = ''

        try:
            font_attrs += FONT_SIZE_ATTR.format(font_sizes[index])
        except (IndexError, TypeError):
            pass

        try:
            font_attrs += FONT_COLOR_ATTR.format(font_colors[index])
        except (IndexError, TypeError):
            pass

        label += FONT_OPEN_TAG.format(font_attrs)
        label += arg
        label += FONT_CLOSE_TAG
        label += '<br/>' if index != len(args) - 1 else ''

    label += '>'
    return label


def to_graphviz(ha: StochasticHybridAutomaton, name: str, SAVE_PATH: str, view=False):
    f = Digraph('hybrid_automaton', filename=SAVE_PATH + name)
    f.attr(rankdir='LR', size='2')
    f.attr('node', shape='circle')
    f.attr('node', fontname='helvetica')
    f.attr('node', penwidth='0.5')
    f.attr('edge', fontname='helvetica')
    f.attr('edge', penwidth='0.5')

    locations = ha.locations
    edges = ha.edges

    for loc in locations:
        if loc.flow_cond is not None:
            label = style_label([loc.name, '<br/><b>' + loc.flow_cond + '</b>'], [8, 6], ['black', '#ad0c00'])
        else:
            label = style_label([loc.name], [8, 6], ['black', '#ad0c00'])
        f.node(loc.name, label=label)

    for edge in edges:
        if edge.guard != '' and edge.sync != '':
            label = style_label([edge.guard, edge.sync], [8, 8], ['#008c05', '#0067b0'])
        elif edge.guard != '':
            label = style_label([edge.guard], [8], ['#008c05'])
        elif edge.sync != '':
            label = style_label([edge.sync], [8], ['#0067b0'])
        else:
            label = ''

        f.edge(edge.start.name, edge.dest.name, label=label)

    f.edge_attr.update(arrowsize='0.5')

    if view:
        f.view()

    return f
