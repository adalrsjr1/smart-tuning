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
from hdrh.histogram import HdrHistogram
import pandas as pd
MONGO_PATH = '/Users/adalbertoibm.com/Coding/Dockerized_AcmeAir/smart-tuning/optimization/k8s-automation/experiments/'

def mongo_tuning(folder, filename):
    metrics = []
    starts = []
    hits = []
    classification = []
    configs = []
    contents = []
    with open(MONGO_PATH + folder + '/' + filename) as f:
        for item in f:
            item = json.loads(item)
            metrics.append(item['metric'])
            starts.append(item['start'])
            hits.append(item['hits'])
            classification.append(item['classification'])
            configs.append(item['configuration'])
            contents.append(item['content'])

    return metrics, starts, hits, classification, configs, contents

def plot_tuning(xs, ys, types, title):
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
    print(ys)
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


    ax.set_title(title)
    ax.set_ylabel('req/s')
    ax.set_xlabel('time (s)')
    ax.set_ylim(ymin=0, ymax=1000)
    ax.legend(title='groups')

    ax.xaxis.set_ticks(xs)
    ax.xaxis.set_ticklabels(xs, rotation='45', fontsize=8)

    ax.set_xticklabels(xs)
    plt.show()
    # plt.savefig(title)
def mongo_perf(folder, filename):
    prod = []
    tuning = []
    with open(MONGO_PATH + folder + '/' + filename) as f:
        for item in f:
            item = json.loads(item)
            prod.append(item['prod_metric'])
            tuning.append(item['tuning_metric'])

    return prod, tuning

def plot_comparison(xs, prod, tuning, types, title):
    fig: Figure = None
    ax: axes.Axes = None
    fig, ax = plt.subplots()

    xs = [xs[i] - xs[0] for i, _ in enumerate(xs)]

    data_y1 = {}
    data_y2 = {}
    data_x = {}

    for t in types:
        if not t in data_x:
            data_x[t] = [0 for _ in xs]
        if not t in data_y1:
            data_y1[t] = [float('NaN') for _ in prod]
        if not t in data_y2:
            data_y2[t] = [float('NaN') for _ in tuning]

    i = 0
    plt.step(xs, prod, 'r-', linewidth=0.7, label='production')
    plt.step(xs, tuning, 'k--', linewidth=0.7, label='training')
    # for x, y, t in zip(xs, prod, types):
    #     data_y1[t][i] = y
    #     # if i + 1 < len(ys):
    #     # data_y[t][i+1] = ys[i+1]
    #     for key in data_x.keys():
    #         data_x[key][i] = x
    #     i += 1
    # for key in data_x.keys():
    #     line, = plt.step(data_x[key], data_y1[key])
    #     line.set_label(key[:6])
    # i = 0
    # for x, y, t in zip(xs, tuning, types):
    #     data_y2[t][i] = y
    #     # if i + 1 < len(ys):
    #     # data_y[t][i+1] = ys[i+1]
    #     for key in data_x.keys():
    #         data_x[key][i] = x
    #     i += 1
    # for key in data_x.keys():
    #     line, = plt.step(data_x[key], data_y2[key])
    #     line.set_label(key[:6])

    ax.set_title(title)
    ax.set_ylabel('req/s')
    ax.set_xlabel('time (s)')
    ax.set_ylim(ymin=0, ymax=1000)
    ax.xaxis.set_ticks(xs)
    ax.xaxis.set_ticklabels(xs, rotation='45', fontsize=8)
    ax.legend(title='pods')
    ax.set_xticklabels(xs)
    plt.show()
    # plt.savefig(title)

def smarttuning_plots():

    folders = ['volume_20200508-2200',
               'volume_20200511-1600',
               'volume_20200511-0800',
               'volume_20200511-1200']
    titles = ['threshold 0% -- sample 20%',
              'threshold 0% -- sample 100%',
              'threshold 10% -- sample 20%',
              'threshold 10% -- sample 100%'
              ]

    for folder, title in zip(folders, titles):

        metrics, starts, _, classification, _, _ = mongo_tuning(folder, 'mongo_tuning.json')
        plot_tuning(starts, metrics, classification, 'Training Pod\n'+title)
        # prod, tuning = mongo_perf(folder, 'mongo_metrics.json')
        # plot_comparison(starts, prod, tuning, classification, title)


def jmeter(folder, filename):
    df = pd.read_csv(MONGO_PATH + folder + '/' + filename, na_values='-')
    pd.set_option('display.max_columns', None)
    df.fillna(0, inplace=True)

    sum = 0
    for key in df.keys():
        if key != 'Elapsed time' and key != 'Gauges Activator' and key != 'Debug Sampler':
            sum += df[key]
    df = pd.DataFrame(sum)
    df.columns = ['Requests']
    return df

def jmeter_plot(df:pd.DataFrame):
    # histogram = HdrHistogram(1, int(max(df.values)[0]+1), 4)
    #
    # for value in df.values:
    #     histogram.record_value(value)
    #
    # encoded_histogram = histogram.encode()
    # print(encoded_histogram)

    df.plot()

    plt.show()

def jmeter_plots():
    folders = ['volume_20200508-2200', 'volume_20200511-0800', 'volume_20200511-1200', 'volume_20200511-1600']

from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
if __name__ == '__main__':
    smarttuning_plots()
