import numpy as np

import config
import seqkmeans
import timeinterval
from util import PrometheusAccessLayer


def mock_sampling():
    with open('letter-recognition.data') as f:
        for line in f:
            v = []
            for i, c in enumerate(line.split(',')):
                if i > 0:
                    v.append(int(c))
            _min = min(v)
            _max = max(v)
            v = np.array(v)
            v = (v - _min) / (_max - _min)
            yield seqkmeans.Container(label=line[0], content_labels=[*range(len(v))], content=v, metric=0)
generator = mock_sampling()

def throughput(namespace, interval, mock=False) -> float:
    """
    query throughput of the applicatoion in a given interval
    :param namespace: application namespace
    :param interval: query for throughput in the past interval (uses module 'timeinterval')
    :return: throughput (req/s)
    """
    print('sampling throughput')
    if mock:
        return np.random.randint(10, 101)

    prometheus = PrometheusAccessLayer(config.PROMETHEUS_ADDR, config.PROMETHEUS_PORT)
    result = prometheus.query(f'sum(rate(remap_http_requests_total{{namespace="{namespace}"}}[{timeinterval.second(interval)}s]))')
    print('\tthroughput: ', result)
    if result.value():
        return float(result.value()[0][1])
    return float('NaN')

def workload(namespace, interval, mock=False) -> seqkmeans.Container:
    """
    query urls requests distribution in a given interval
    :param namespace: application namespace
    :param interval: query for throughput in the past interval (uses module 'timeinterval')
    :return: Workload object. Note that urls are grouped in case of using <Path Parameters>, e.g.,

    the urls:

    /my/url/using/path-parameter/uid153@email.com
    /my/url/using/path-parameter/uid188@email.com
    /my/url/using/path-parameter/uid174@email.com

    are grouped into /my/url/using/path-parameter/uid153@email.com

    """
    print('sampling urls')
    if mock:
        return next(generator)

    prometheus = PrometheusAccessLayer(config.PROMETHEUS_ADDR, config.PROMETHEUS_PORT)
    result = prometheus.query(f'sum by (path)(rate(remap_http_requests_total{{namespace="{namespace}"}}[{timeinterval.second(interval)}s])) / ignoring '
                              f'(path) group_left sum(rate(remap_http_requests_total{{namespace="{namespace}"}}[{timeinterval.second(interval)}s]))')

    urls = [u['path'] for u in result.metric()]
    values = [float(u[1]) for u in result.value()]

    if urls and values:
        urls, values = zip(*sorted(zip(urls, values)))

    group = __compare__(urls, 0.98)

    match = {}
    for key, items in group.items():
        match.update({urls[key] : sum([values[k] for k in items])})
    print('\turls: ', match)
    return seqkmeans.Container(label=str(timeinterval.now()), content_labels=list(match.keys()), content=np.array(list(match.values())), metric=0)

def __distance__(u, v):
    len_u = len(u)
    len_v = len(v)
    u = np.array([ord(c) for c in u] + [0] * (max(len_u, len_v) - len_u) )
    v = np.array([ord(c) for c in v] + [0] * (max(len_u, len_v) - len_v) )

    # cosine distance
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # euclidean distance
    # return np.linalg.norm(u-v)

def __compare__(histograms, threshould=0):
    from collections import defaultdict
    workflows_group = defaultdict(set)
    memory = set()
    for i, hist1 in enumerate(histograms):
        for j, hist2 in enumerate(histograms):
            d = __distance__(hist1, hist2)
            if d >= threshould:
                __group__(i, j, workflows_group, memory)

    return workflows_group


# TODO: optimize this in the future
def __group__(u, v, table, memory):
    if u not in memory and v not in memory:
        table[u].add(v)
    elif u in memory and v not in memory:
        if u in table:
            table[u].add(v)
        else:
            return __group__(v, u, table, memory)
    elif u not in memory and v in memory:
        for key, value in table.items():
            if v in value:
                value.add(u)
                break
    memory.add(u)
    memory.add(v)


def workload_and_metric(namespace, interval, mock=False) -> seqkmeans.Container:
    print('merging urls and throughput')
    with config.ThreadPoolExecutor(2) as executor:
        future_workload = executor.submit(workload, config.NAMESPACE, config.WAITING_TIME, mock)
        future_throughput = executor.submit(throughput, config.NAMESPACE, config.WAITING_TIME, mock)
        done = config.ThreadWait([future_workload, future_throughput], timeout=None, return_when=config.FUTURE_ALL_COMPLETED)

        future_throughput = done[0].pop().result()
        future_workload = done[0].pop().result()
        if not isinstance(future_workload, seqkmeans.Container):
            future_workload, future_throughput = future_throughput, future_workload

    _workload = future_workload
    _workload.metric = future_throughput

    return _workload

classificationCtx = seqkmeans.KmeansContext(config.K)
def classify(workload:seqkmeans.Container) -> seqkmeans.Cluster:
    print('classifying workload ', workload.label)
    type, hits = classificationCtx.cluster(workload)
    print(f'workload {workload.label} classified as {type.id} -- {hits}th hit')
    return type, hits
