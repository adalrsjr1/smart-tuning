from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl

from util import PrometheusAccessLayer
import timeinterval as _time

PROMETHEUS_ADDR = 'localhost'
PROMETHEUS_PORT = '30090'
NAMESPACE = 'default'
TIME_INTERVAL = _time.minute(40)

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

prometheus = PrometheusAccessLayer(PROMETHEUS_ADDR, PROMETHEUS_PORT)
def app_histogram(timeinterval):

    result = prometheus.query(
        f'sum by (path)(rate(remap_http_requests_total{{namespace="{NAMESPACE}"}}[{timeinterval}s])) / ignoring '
        f'(path) group_left sum(rate(remap_http_requests_total{{namespace="{NAMESPACE}"}}[{timeinterval}s]))')

    urls = [u['path'] for u in result.metric()]
    values = [float(u[1]) for u in result.value()]

    urls, values = zip(*sorted(zip(urls, values)))

    group = __compare__(urls, 0.98)

    histogram = {}
    for key, items in group.items():
        histogram.update({urls[key]: sum([values[k] for k in items])})

    return histogram

def app_metrics(timeinterval):
    # result = prometheus.query(f'sum(rate(remap_http_requests_total{{namespace="{NAMESPACE}"}}[{timeinterval}s]))')
    result = prometheus.query(f'remap_http_requests_total{{namespace="{NAMESPACE}"}}')
    return float(result.value()[0][1])


if __name__ == '__main__':
    # mpl.style.use('grayscale')
    histogram = app_histogram(TIME_INTERVAL)
    fig, ax = plt.subplots()

    xs, ys = np.arange(0, len(histogram)), list(histogram.values())
    print(xs)
    print(ys)
    barlist = plt.bar(xs, ys)

    # for i, bar in enumerate(barlist):
    #     bar.set_color(f'C{i}')


    ax.set_ylabel('Probabilities 0.0 - 1.0')
    ax.set_title('Distribution of requests')

    ax.set_ylim([0, 1])
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    ax.set_xticks(xs)

    labels = [f'url-{x}' for x in xs]
    ax.set_xticklabels(labels)

    plt.show()


