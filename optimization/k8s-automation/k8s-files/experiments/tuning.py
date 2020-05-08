from __future__ import annotations
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D


MONGO_PATH = '/Users/adalbertoibm.com/Coding/Dockerized_AcmeAir/smart-tuning/optimization/k8s-automation/experiments/volume/'

def mongo_tuning(filename):
    metrics = []
    starts = []
    hits = []
    classification = []
    configs = []
    contents = []
    with open(MONGO_PATH+filename) as f:
        for item in f:
            item = json.loads(item)
            metrics.append(item['metric'])
            starts.append(item['start'])
            hits.append(item['hits'])
            classification.append(item['classification'])
            configs.append(item['configuration'])
            contents.append(item['content'])

    return metrics, starts, hits, classification, configs, contents

def plot_mongo(xs, ys, types):
    fig:Figure = None
    ax:axes.Axes = None
    fig, ax = plt.subplots()

    xs = [xs[i] - xs[0] for i, _ in enumerate(xs)]

    data_y = {}
    data_x = {}

    for t in types:
        if not t in data_x:
            data_x[t] = [0 for _ in xs]
        if not t in data_y:
            data_y[t] = [float('NaN') for _ in ys]

    i = 0
    plt.step(xs, ys, 'k--', linewidth=.5)
    for x, y, t in zip(xs, ys, types):
        data_y[t][i] = y
        # if i + 1 < len(ys):
            # data_y[t][i+1] = ys[i+1]
        for key in data_x.keys():
            data_x[key][i] = x
        i += 1
    for key in data_x.keys():
        line, = plt.step(data_x[key], data_y[key])
        line.set_label(key[:6])


    ax.set_title('Throughput')
    ax.set_ylabel('req/s')
    ax.set_xlabel('time (s)')

    ax.legend(title='groups')

    ax.xaxis.set_ticks(xs)
    ax.xaxis.set_ticklabels(xs, rotation='45', fontsize=8)

    ax.set_xticklabels(xs)

    plt.show()

def mongo_perf(filename):
    prod = []
    tuning = []
    with open(MONGO_PATH+filename) as f:
        for item in f:
            item = json.loads(item)
            prod.append(item['prod_metric'])
            tuning.append(item['tuning_metric'])

    return prod, tuning




if __name__ == '__main__':
    metrics, starts, _, classification, _, _ = mongo_tuning('mongo_tuning.json')
    # plot_mongo(starts, metrics, classification)
    prod, tuning = mongo_perf('mongo_perf.json')
    plot_mongo(starts, prod, classification)