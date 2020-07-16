import sampler
import os, sys
import kubernetes
from collections import defaultdict
import pandas as pd
import math
import networkx as nx
import re
pd.set_option('display.max_columns', None)
from kubernetes.client.models import *
from prometheus_pandas import query as handler
interval = 900

podname = 'acmeair.*smarttuning-'
# latency
q3 = f'avg( rate(smarttuning_http_processtime_seconds_sum{{pod=~"{podname}.*",name!~".*POD.*"}}[{interval}s])) by (pod, src, dst, instance, service) /' \
    f'avg( rate(smarttuning_http_processtime_seconds_count{{pod=~"{podname}.*",name!~".*POD.*"}}[{interval}s])) by (pod, src, dst, instance, service)'

# throughput
q1 = f'avg( rate(in_http_requests_total{{pod=~"{podname}.*",name!~".*POD.*"}}[{interval}s])) by (pod, src, dst, instance, service) /' \
     f'avg( rate(out_http_requests_total{{pod=~"{podname}.*",name!~".*POD.*"}}[{interval}s])) by (pod, src, dst, instance, service)'
q2 = f'avg(rate(smarttuning_http_requests_total{{pod=~"{podname}.*",name!~".*POD.*"}}[{interval}s])) by (pod, src, dst, instance, service)'
# memory
# q2 = f'quantile(1, quantile_over_time(1,container_memory_working_set_bytes{{pod=~"{podname}.*",name!~".*POD.*"}}[{interval}s])) (pod)'

t = None
timeout = None
p = handler.Prometheus('http://localhost:8001')
params = {'query': q1}
params.update({'time': t} if t is not None else {})
params.update({'timeout': timeout.total_seconds()} if timeout is not None else {})

# sample metrics
result = p._do_query('/api/v1/namespaces/kube-monitoring/services/prometheus-service/proxy/api/v1/query', params)

table = {}
requests_df:pd.DataFrame = sampler.series_to_dataframe(handler.to_pandas(result))
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

print(links_df)

# transform table into graph
G = nx.MultiDiGraph()
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
        G.add_edge(row['src'], row['dst'], weight=-v, label=f'[{-v:.2f}]')

# remove cycles
gateway = ''
for degree in G.in_degree():
    print(degree)
    if degree[1] == 0:
        gateway = degree[0]
        break

# print('degree', G.degree(weight='weight'))
# print()
if not gateway:
    raise Exception('gateway is null')
# print(gateway)

while True:
    shortest = ()
    shortest_value = float('inf')
    try:
        for edge in nx.find_cycle(G, gateway):
            value = G.get_edge_data(*edge, default={'weight':0}).get('weight', 0)
            if not shortest or value > shortest_value:
                shortest = edge
                shortest_value = value
        G.remove_edge(*shortest)
    except nx.exception.NetworkXNoCycle:
        break

longest_path = nx.algorithms.dag.dag_longest_path(G, weight='weight', default_weight=0)
# print(longest_path, nx.algorithms.dag_longest_path_length(G, weight='weight', default_weight=0))
from pprint import pprint
lenght, path = nx.algorithms.shortest_paths.weighted.single_source_bellman_ford(G, source=gateway, weight='weight')

shortest = ()
for key, value in lenght.items():
    if not shortest or value < shortest[1]:
        shortest = (key, value)

# print('lenght', lenght)
# print()
# print('shortest', shortest)
# print()
# print('path', path)
# print()
print('best', path[shortest[0]])
nx.drawing.nx_pydot.write_dot(G, 'graph.dot')


# workload C: best ['acmeair-nginx-servicesmarttuning', 'acmeair-booking-servicesmarttuning', 'acmeair-flight-servicesmarttuning']
# workload B: best ['acmeair-nginx-servicesmarttuning', 'acmeair-booking-servicesmarttuning', 'acmeair-flight-servicesmarttuning']
# workload A: best ['acmeair-nginx-servicesmarttuning', 'acmeair-booking-servicesmarttuning', 'acmeair-flight-servicesmarttuning']


