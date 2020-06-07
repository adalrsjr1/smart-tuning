from __future__ import annotations
import numpy as np
from concurrent.futures import Future
import config
import seqkmeans
import timeinterval

from prometheus_pandas import query as handler

_prometheus = handler.Prometheus(f'http://{config.PROMETHEUS_ADDR}:{config.PROMETHEUS_PORT}')

def do_sample(query:str, endpoint=_prometheus) -> Future:
    return config.executor.submit(endpoint.query, query)

def throughput(pod_regex, interval, quantile=1.0, endpoint=_prometheus) -> Future:
    """ return a concurrent.futures.Future<pandas.Series> with the total_requests rate of an specific pod"""
    print(f' >>> sampling throughput at {pod_regex}')
    query = f'quantile({quantile},rate(smarttuning_http_requests_total{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s]))'

    return do_sample(query, endpoint=endpoint)

def latency(pod_regex, interval, quantile=1.0, endpoint=_prometheus) -> Future:
    """ return a concurrent.futures.Future<pandas.Series> with the latency_sum/latency_count rate of an specific pod"""
    print(f' >>> sampling latency at {pod_regex}')
    query = f'quantile({quantile}, rate(smarttuning_http_latency_seconds_sum{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s])) / ' \
    f'quantile({quantile}, rate(smarttuning_http_latency_seconds_count{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s]))'

    return do_sample(query, endpoint=endpoint)

def memory(pod_regex, interval, quantile=1.0, endpoint=_prometheus) -> Future:
    """ return a concurrent.futures.Future<pandas.Series> with the memory (bytes) quantile over time of an specific pod
        :param quantile a value 0.0 - 1.0
    """
    print(f' >>> sampling memory at {pod_regex}')
    query = f'quantile({quantile}, quantile_over_time({quantile},container_memory_working_set_bytes{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s]))'

    return do_sample(query, endpoint=endpoint)

def cpu(pod_regex, interval, quantile=1.0, endpoint=_prometheus) -> Future:
    """ return a concurrent.futures.Future<pandas.Series> with the CPU (milicores) rate over time of an specific pod
    """
    print(f' >>> sampling cpu at {pod_regex}')
    query = f'quantile({quantile},rate(container_cpu_system_seconds_total{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s]))'
    return do_sample(query, endpoint=endpoint)

def workload(pod_regex, interval, endpoint=_prometheus) -> Future:
    """
    query urls requests distribution in a given interval
    :param pod_regex: application namespace
    :param interval: query for throughput in the past interval (uses module 'timeinterval')
    :return: Workload object. Note that urls are grouped in case of using <Path Parameters>, e.g.,

    the urls:

    /my/url/using/path-parameter/uid153@email.com
    /my/url/using/path-parameter/uid188@email.com
    /my/url/using/path-parameter/uid174@email.com

    are grouped into /my/url/using/path-parameter/uid153@email.com

    """
    print(f' >>> sampling urls at {pod_regex}')

    query = f'sum by (path)(rate(smarttuning_http_requests_total{{pod=~"{pod_regex}"}}[{timeinterval.second(interval)}s]))' \
            f' / ignoring ' \
            f'(path) group_left sum(rate(smarttuning_http_requests_total{{pod=~"{pod_regex}"}}[{timeinterval.second(interval)}s]))'

    return do_sample(query, endpoint=endpoint)

    # urls = [u['path'] for u in result.metric()]
    # values = [float(u[1]) for u in result.value()]
    #
    # # sort histogram by urls
    # if urls and values:
    #     urls, values = zip(*sorted(zip(urls, values)))
    #
    # # group urls by similarity, e.g., api/v1/id3 and api/v1/id57 should be grouped
    # group = __compare__(urls, 0.98)
    #
    # match = {}
    # for key, items in group.items():
    #     match.update({urls[key] : sum([values[k] for k in items])})
    # return seqkmeans.Container(label=str(timeinterval.now()), content_labels=list(match.keys()), content=np.array(list(match.values())), metric=0)
# #
# # def __distance__(u, v):
# #     len_u = len(u)
# #     len_v = len(v)
# #     u = np.array([ord(c) for c in u] + [0] * (max(len_u, len_v) - len_u) )
# #     v = np.array([ord(c) for c in v] + [0] * (max(len_u, len_v) - len_v) )
# #
# #     # cosine distance
# #     return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
# #     # euclidean distance
# #     # return np.linalg.norm(u-v)
# #
# # from collections import defaultdict
# # def __compare__(histograms, threshould=0):
# #     workflows_group = defaultdict(set)
# #     memory = set()
# #     for i, hist1 in enumerate(histograms):
# #         for j, hist2 in enumerate(histograms):
# #             d = __distance__(hist1, hist2)
# #             if d >= threshould:
# #                 __group__(i, j, workflows_group, memory)
# #
# #     return workflows_group
# #
#
# # TODO: optimize this in the future
# def __group__(u, v, table, memory):
#     if u not in memory and v not in memory:
#         table[u].add(v)
#     elif u in memory and v not in memory:
#         if u in table:
#             table[u].add(v)
#         else:
#             return __group__(v, u, table, memory)
#     elif u not in memory and v in memory:
#         for key, value in table.items():
#             if v in value:
#                 value.add(u)
#                 break
#     memory.add(u)
#     memory.add(v)
#
#
# def workload_and_metric(pod_regex, interval, mock=False) -> seqkmeans.Container:
#     future_workload = config.executor.submit(workload, pod_regex, interval, mock)
#     future_throughput = config.executor.submit(throughput, pod_regex, interval, mock)
#     done = config.ThreadWait([future_workload, future_throughput], timeout=None, return_when=config.FUTURE_ALL_COMPLETED)
#
#     future_throughput = done[0].pop().result()
#     future_workload = done[0].pop().result()
#     if not isinstance(future_workload, seqkmeans.Container):
#         future_workload, future_throughput = future_throughput, future_workload
#
#     _workload = future_workload
#     _workload.metric = future_throughput
#     _workload.end = timeinterval.now()
#     _workload.start =  _workload.end - interval
#
#     return _workload
