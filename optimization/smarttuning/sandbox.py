import sampler
import os, sys
import kubernetes
from collections import defaultdict
import pandas as pd
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from kubernetes.client.models import *
from prometheus_pandas import query as handler
import time
step = 7200
#
# G = nx.Graph()
# G.add_weighted_edges_from([
# ('0', '1', 0),
# ('0', '2', 0),
# ('0', '3', 0),
# ('0', '4', 0),
# ('0', '5', 0),
# ('0', '6', 0),
# ('6', '1', 0.13),
# ('6', '2', 0.12),
# ('6', '3', 0.14),
# ('6', '4', 0.24),
# ('6', '5', 0),
# ('1', '3', 0.45),
# ('2', '3', 0.44),
# ('2', '4', 0.48),
# ])
#
# pr = nx.algorithms.link_analysis.pagerank(G, weight='weight')
# rank_vector=np.array([[*pr.values()]])
# best_node=np.argmax(rank_vector)
# print(pr, best_node)
#
# G.add_weighted_edges_from([
# ('1', '6', 0.13),
# ('2', '6', 0.12),
# ('3', '6', 0.14),
# ('4', '6', 0.24),
# ('5', '6', 0),
# ('3', '1', 0.45),
# ('3', '2', 0.44),
# ('4', '2', 0.48),
# ])
#
# pr = nx.algorithms.link_analysis.pagerank(G)
# rank_vector=np.array([[*pr.values()]])
# best_node=np.argmax(rank_vector)
# print(pr, best_node)
#
#
#
#
# sys.exit(0)
podname = '.*service-.*'
# throughput
q1 = f'sum( count_over_time(in_http_requests_total{{pod=~"{podname}",name!~".*POD.*"}}[{step}s])) by (pod, src, dst, instance, service) /' \
     f'sum( count_over_time(out_http_requests_total{{pod=~"{podname}",name!~".*POD.*"}}[{step}s])) by (pod, src, dst, instance, service) '

q1 = f'sum(count_over_time(smarttuning_http_requests_total{{pod=~"{podname}",name!~".*POD.*"}}[{step}s])) by (pod, src, dst, instance, service)'

q1 = f'(sum(count_over_time(smarttuning_http_requests_total{{pod=~"{podname}",name!~".*POD.*"}}[{step}s])) by (src, dst, instance, service, pod)) / ' \
     f'ignoring(src, dst, instance, service, pod) group_left sum(count_over_time(smarttuning_http_requests_total{{pod=~"{podname}",name!~".*POD.*"}}[{step}s]))'

# t = None
# timeout = None
p = handler.Prometheus('http://trxrhel7perf-1.canlab.ibm.com:30099')
result = p.query(q1)
now = time.time()
result = p.query_range(q1, now-(3600*2), now, step)
# result.to_csv('data-202007221834.csv')

# result = pd.read_csv('data-202007221834.csv')
# result = pd.read_csv('data-202007221347.csv')

# print(result)
# sys.exit(0)
new_result = result.T
for _i, _item in enumerate(new_result.iteritems()):
    result = _item[1]

    table = {}
    # requests_df:pd.DataFrame = sampler.series_to_dataframe(handler.to_pandas(result))
    requests_df:pd.DataFrame = sampler.series_to_dataframe(result)

    links_df = requests_df.copy()

    # clean dataframe and create hash table
    for i, item in links_df.iterrows():
        splitted_ip = item['instance'].split(':')[0]
        links_df.loc[i, ('instance')] = splitted_ip
        table[splitted_ip] = item['service']


    # replace IPs with service names
    for i, instance in enumerate(links_df['instance']):
        links_df.loc[i, ('dst')] = table.get(links_df.loc[i, ('dst')], links_df.loc[i, ('dst')])
        links_df.loc[i, ('src')] = table.get(links_df.loc[i, ('src')], links_df.loc[i, ('src')])
        # for j, row in links_df.iterrows():
        #     links_df.loc[j, ('dst')] = row['service']
        #
        #     if row['src'] == instance:
        #         links_df.loc[j, ('src')] = links_df.loc[i, ('service')]

    # transform table into graph
    # G = nx.MultiDiGraph()
    G = nx.DiGraph()
    ip_regex = re.compile("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    for i, row in links_df.iterrows():
        if not ip_regex.match(row['src']) and not ip_regex.match(row['dst']):
            v = float(row['value'])
            if math.isnan(v):
                continue
            try:
                path = row["path"]
            except KeyError:
                path = ''
            if G.has_edge(row['src'], row['dst']):
                weight = G[row['src']][row['dst']]['weight']
                G[row['src']][row['dst']]['weight'] += v
                G[row['src']][row['dst']]['label'] = f"[{G[row['src']][row['dst']]['weight']:.5f}]"
            else:
                G.add_edge(row['src'], row['dst'], weight=1, label=f'[{v:.5f}]')

    # remove cycles
    gateway = ''
    for degree in G.in_degree():
        if degree[1] == 0:
            gateway = degree[0]
            break

    # print('degree', G.degree(weight='weight'))
    # print()
    # if not gateway:
        # raise Exception('gateway is null')
        # print('gateway is null')
        # continue
    # print(gateway)

    if gateway:
        for edge in G.edges:
            if edge[0] == gateway:
                if not G.has_edge(edge[1], gateway):
                    G.add_edge(edge[1], gateway, weight=1)
            # G.add_edge(node, gateway, weight=float('1'))
    # while True:
    #     shortest = ()
    #     shortest_value = float('inf')
    #     try:
    #         for edge in nx.find_cycle(G, gateway):
    #             value = G.get_edge_data(*edge, default={'weight':0}).get('weight', 0)
    #             if not shortest or value > shortest_value:
    #                 shortest = edge
    #                 shortest_value = value
    #         G.remove_edge(*shortest)
    #     except nx.exception.NetworkXNoCycle:
    #         break
    #
    # longest_path = nx.algorithms.dag.dag_longest_path(G, weight='weight', default_weight=0)
    # lenght, path = nx.algorithms.shortest_paths.weighted.single_source_bellman_ford(G, source=gateway, weight='weight')
    #
    # shortest = ()
    # for key, value in lenght.items():
    #     if not shortest or value < shortest[1]:
    #         shortest = (key, value)

    # print(f'best {_i}', path[shortest[0]])
    nx.drawing.nx_pydot.write_dot(G, f'graph_{_i:02}.dot')
    pr = nx.algorithms.link_analysis.pagerank(G, weight='weight')
    rank_vector=np.array([[*pr.values()]])
    best_node=np.argmax(rank_vector)
    print(pr, best_node, rank_vector[0,best_node])


