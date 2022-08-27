import yaml
import re
import sys
from yaml.loader import SafeLoader
import pygraphviz as pgv
from pprint import pprint

g = pgv.AGraph(directed=True)


def clean_arithmetic(node_name: str):
    tokens = node_name.split(' ')
    return [token for token in tokens if re.compile('[-a-zA-Z]+').match(token)][0]


if '__main__' == __name__:
    dependencies = []

    with open('./daytrader-ss.yaml') as file:
        search_space = yaml.load(file, Loader=SafeLoader)

        for item in search_space['data']:
            for tunable_type in item['tunables']:
                for node in item['tunables'][tunable_type]:
                    try:

                        if 'virtualized' in node['name'] or \
                           'gc' in node['name'] or \
                           'contaier_support' in node['name']:
                            continue

                        g.add_node(node['name'])
                        if 'dependsOn' in node['lower']:
                            parent = clean_arithmetic(node['lower']['dependsOn'])
                            g.add_node(parent)
                            dependencies.append((node['name'], parent))
                        if 'dependsOn' in node['upper']:
                            parent = clean_arithmetic(node['upper']['dependsOn'])
                            dependencies.append((node['name'], parent))
                    except KeyError:
                        g.add_node(node['name'])

    for dependency in dependencies:
        g.add_edge(dependency[0], dependency[1])

    g.draw('dependency.svg', prog='dot')
