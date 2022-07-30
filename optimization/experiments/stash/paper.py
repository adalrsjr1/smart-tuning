import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib as mpl
import matplotlib
import hashlib
import math
import sys
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FixedLocator)

import matplotlib.lines as mlines
from matplotlib.text import Annotation



def load_data(filename:str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df.fillna(0)

def split_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    :param table:
    :return: metric and workload table
    """
    # metrics
    metrics_table = table[[
        'production_metric.cpu',
        'production_metric.memory',
        'production_metric.throughput',
        'production_metric.process_time',
        'production_metric.errors',
        'production_metric.objective',
        'training_metric.cpu',
        'training_metric.memory',
        'training_metric.throughput',
        'training_metric.process_time',
        'training_metric.errors',
        'training_metric.objective',
        'tuned'
    ]].copy(),
    # workloads
    workload_table = table[[
        'production_workload.path',
        'production_workload.value',
        'training_workload.path',
        'training_workload.value',
        'tuned'
    ]].copy()

    # configs
    configs_table = None
    try:
        configs_table = table[[
            'last_config.daytrader-service.cpu',
            'last_config.daytrader-service.memory',
            'last_config.daytrader-config-app.CONMGR1_MAX_POOL_SIZE',
            'last_config.daytrader-config-app.HTTP_MAX_KEEP_ALIVE_REQUESTS',
            'last_config.daytrader-config-app.MAX_THREADS',
            'production_metric.objective',
        ]]
    except:
        print("none last config")

    metrics_table = metrics_table[0]
    # row0 = metrics_table.iloc[[0]]
    # row0.loc[0, 'training_metric.cpu'] = None
    # row0.loc[0, 'training_metric.memory'] = None
    # row0.loc[0, 'training_metric.throughput'] = None
    # row0.loc[0, 'training_metric.process_time'] = None
    # row0.loc[0, 'training_metric.errors'] = None
    # row0.loc[0, 'training_metric.objective'] = 0
    # row0.loc[0, 'tuning'] = False
    # metrics_table = row0.iloc[[0]].append(metrics_table, ignore_index=False)
    #
    # rowN = metrics_table.iloc[[-1]]
    # print(rowN)
    # rowN.loc[0, 'training_metric.cpu'] = None
    # rowN.loc[0, 'training_metric.memory'] = None
    # rowN.loc[0, 'training_metric.throughput'] = None
    # rowN.loc[0, 'training_metric.process_time'] = None
    # rowN.loc[0, 'training_metric.errors'] = None
    # rowN.loc[0, 'training_metric.objective'] = 0
    #
    # # metrics_table = metrics_table.append(rowN, ignore_index=False)


    return metrics_table, workload_table, configs_table

def plot_metrics(table:pd.DataFrame, title:str):
    nmetrics = 6
    nrows = len(table.index)
    fig, axs = plt.subplots(nrows=nmetrics, ncols=1, figsize=(12, 8), sharex='col', gridspec_kw = {'wspace':0, 'hspace':0.5})
    target = [axs[0], axs[1], axs[2], axs[3], axs[4], axs[5]]

    table['production_metric.objective'] *= -1
    table['training_metric.objective'] *= -1
    table['production_metric.memory'] /= 2**20
    table['training_metric.memory'] /= 2**20
    table['tuned'] = table['tuned'].apply(lambda row: 0 if row == 'false' or row == False else 1)

    table[[
        'production_metric.cpu',
        'production_metric.memory',
        'production_metric.throughput',
        'production_metric.process_time',
        'production_metric.errors',
        'production_metric.objective']].plot(ax=target, linewidth=1, drawstyle='steps-post', style=['b-']*nmetrics, subplots=True)

    table[[
        'training_metric.cpu',
        'training_metric.memory',
        'training_metric.throughput',
        'training_metric.process_time',
        'training_metric.errors',
        'training_metric.objective']].plot(ax=target, linewidth=1, drawstyle='steps-post', style=['r--']*nmetrics, subplots=True)

    # table[['tuned']*nmetrics].plot(ax=target, linewidth=0.7, style=['k:']*nmetrics, subplots=True, )

    for it, istuned in enumerate(table['tuned']):
        if istuned:
            newline_yspan([it + 1, 0], [it + 1, 10], axs[0])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[1])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[2])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[3])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[4])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[5])

    maxt = max(table['training_metric.cpu'])
    maxp = max(table['production_metric.cpu'])
    axs[0].set_yticks(np.arange(0, max(maxt, maxp) + 1, max(maxt, maxp) / 4))
    # axs[0].xaxis.set_ticks([0,2,4,8,16])
    axs[0].set_xlim([0, nrows - 1])
    # ax2 = axs[0].twinx()
    axs[0].set_ylabel('cpu\n(cores)')
    axs[0].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[0].yaxis.set_ticks_position('both')

    # ax2.set_yticks(axs[0].get_yticks())

    # set memory axis

    maxt = max(table['training_metric.memory'])
    maxp = max(table['production_metric.memory'])
    axs[1].set_yticks(np.arange(0, max(maxt, maxp) + 1, max(maxt, maxp) / 4))
    axs[1].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[1].set_xlim([0, nrows - 1])
    # ax2 = axs[1].twinx()
    axs[1].set_ylabel('memory\n(MiB)')
    axs[1].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[1].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[1].get_yticks())
    # ax2.yaxis.set_major_locator(FixedLocator([0.5,2,8]))
    # ax2.yaxis.set_minor_locator(FixedLocator([1,4]))
    # ax2.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    # ax2.tick_params(which='major', pad=20, axis='y')

    # set throughput axis

    maxt = max(table['training_metric.throughput'])
    maxp = max(table['production_metric.throughput'])
    axs[2].set_yticks(np.linspace(0, max(maxt, maxp), 5))
    axs[2].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[2].set_xlim([0, nrows - 1])
    # ax2 = axs[2].twinx()
    axs[2].set_ylabel('rqs')
    axs[2].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[2].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[2].get_yticks())
    # axs[2].set_ylim([0, 10000])

    # set process axis

    maxt = max(table['training_metric.process_time'])
    maxp = max(table['production_metric.process_time'])
    axs[3].set_yticks(np.linspace(0, max(maxt, maxp), 4))
    axs[3].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[3].set_xlim([0, nrows - 1])
    # ax2 = axs[3].twinx()
    axs[3].set_ylabel('resp. time(s)')
    axs[3].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[3].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[3].get_yticks())
    # axs[2].set_ylim([0, 10000])

    # axs[2].set_ylim([0, 10000])
    # print(table['training_metric.process_time'])
    # print(table['production_metric.process_time'])

    # set errors axis
    maxt = max(table['training_metric.errors'])
    maxp = max(table['production_metric.errors'])
    axs[4].set_yticks(np.linspace(0, max(maxt, maxp), 4))
    axs[4].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[4].set_xlim([0, nrows - 1])
    # axs[4].set_ylim([0, 100])
    # ax2 = axs[4].twinx()
    axs[4].set_ylabel('errors (%)')
    axs[4].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[4].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[4].get_yticks())

    # set objective axis

    maxt = max(table['training_metric.objective'])
    maxp = max(table['production_metric.objective'])
    axs[5].set_yticks(np.linspace(0, max(maxt, maxp), 4))
    axs[5].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[5].set_xlim([0, nrows - 1])
    # ax2 = axs[5].twinx()
    axs[5].set_ylabel('requests/$')
    axs[5].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[5].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[5].get_yticks())
    # axs[5].set_ylim([0, max(table['training_metric.objective'])])

    axs[1].legend(frameon=False, )
    for i in range(1,nmetrics):
        axs[i].get_legend().remove()
        axs[i].margins(x=0)

    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(mlines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=10, markeredgewidth=1.5))

    axs[0].legend(handles, ['production', 'training', 'tuning'], bbox_to_anchor=(0., 1.3, 1., .102),
                  loc='upper center',
                  ncol=3, borderaxespad=0., frameon=False, )
    axs[-1].set_xlabel('iterations')
    axs[-1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0].set_title(title, loc='left')

    plt.show()

def plot_metrics_summarized(table:pd.DataFrame, title:str):
    table['production_metric.objective'] *= -1
    table['training_metric.objective'] *= -1
    table['production_metric.memory'] /= 2 ** 20
    table['training_metric.memory'] /= 2 ** 20
    table['production_metric.errors'] *= 100
    table['training_metric.errors'] *= 100
    table['tuned'] = table['tuned'].apply(lambda row: 0 if row == 'false' or row == False else 1)

    # print(table)


    columns = [
        'last_config.daytrader-config-app.CONMGR1_AGED_TIMEOUT',
        'last_config.daytrader-config-app.CONMGR1_MAX_IDLE_TIMEOUT',
        'last_config.daytrader-config-app.CONMGR1_MAX_POOL_SIZE',
        'last_config.daytrader-config-app.CONMGR1_MIN_POOL_SIZE',
        'last_config.daytrader-config-app.CONMGR1_REAP_TIME',
        'last_config.daytrader-config-app.CONMGR1_TIMEOUT',
        'last_config.daytrader-config-app.HTTP_MAX_KEEP_ALIVE_REQUESTS',
        'last_config.daytrader-config-app.HTTP_PERSIST_TIMEOUT',
        'last_config.daytrader-config-app.MAX_THREADS',
        'last_config.daytrader-service.cpu',
        'last_config.daytrader-service.memory',
    ]

    hashes = []
    for index, row in table.iterrows():
        unique = hashlib.md5(bytes(str(tuple(row[columns].values)), 'ascii')).hexdigest()
        # print(unique)
        hashes.append(unique)

    table['config'] = hashes

    grouped_table = table.groupby('config') \
        .agg({
        'production_metric.objective': ['mean', 'min', 'max'],
        'production_metric.throughput': ['mean', 'min', 'max'],
        'production_metric.cpu': ['mean', 'min', 'max'],
        'production_metric.memory': ['mean', 'min', 'max'],
        'production_metric.process_time': ['mean', 'min', 'max'],
        'production_metric.errors': ['mean', 'min', 'max'],
    })
    grouped_table.columns = [
        'production_metric.mean_objective', 'production_metric.min_objective', 'production_metric.max_objective',
        'production_metric.mean_throughput', 'production_metric.min_throughput', 'production_metric.max_throughput',
        'production_metric.mean_cpu', 'production_metric.min_cpu', 'production_metric.max_cpu',
        'production_metric.mean_memory', 'production_metric.min_memory', 'production_metric.max_memory',
        'production_metric.mean_process_time', 'production_metric.min_process_time', 'production_metric.max_process_time',
        'production_metric.mean_errors', 'production_metric.min_errors', 'production_metric.max_errors',
                             ]
    grouped_table = grouped_table.reset_index()

    # print(grouped_table)
    # table = table.groupby((table.config!=table.config.shift()).cumsum()).agg({'production_metric.objective':['mean', 'min', 'max']}).reset_index()
    table = table.merge(grouped_table, on='config', how='left').reset_index()

    nmetrics = 6
    nrows = len(table.index)
    fig, axs = plt.subplots(nrows=nmetrics, ncols=1, figsize=(12, 8), sharex='col', gridspec_kw = {'wspace':0, 'hspace':0.5})
    target = [axs[0], axs[1], axs[2], axs[3], axs[4], axs[5]]


    table[[
        'production_metric.mean_cpu',
        'production_metric.mean_memory',
        'production_metric.mean_throughput',
        'production_metric.mean_process_time',
        'production_metric.mean_errors',
        'production_metric.min_objective']].plot(ax=target, linewidth=1, drawstyle='steps-post', style=['b-']*nmetrics, subplots=True)

    table[[
        'training_metric.cpu',
        'training_metric.memory',
        'training_metric.throughput',
        'training_metric.process_time',
        'training_metric.errors',
        'training_metric.objective']].plot(ax=target, linewidth=1, drawstyle='steps-post', style=['r--']*nmetrics, subplots=True)

    # table[['tuned']*nmetrics].plot(ax=target, linewidth=0.7, style=['k:']*nmetrics, subplots=True, )

    for it, istuned in enumerate(table['tuned']):
        if istuned:
            newline_yspan([it + 1, 0], [it + 1, 10], axs[0])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[1])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[2])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[3])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[4])
            newline_yspan([it + 1, 0], [it + 1, 10], axs[5])

    # set cpu axis

    maxt = max(table['training_metric.cpu'])
    maxp = max(table['production_metric.mean_cpu'])
    axs[0].set_yticks(np.arange(0, max(maxt, maxp)+1, max(maxt, maxp)/4))
    # axs[0].xaxis.set_ticks([0,2,4,8,16])
    axs[0].set_xlim([0, nrows-1])
    # ax2 = axs[0].twinx()
    axs[0].set_ylabel('cpu\n(cores)')
    axs[0].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[0].yaxis.set_ticks_position('both')

    # ax2.set_yticks(axs[0].get_yticks())


    # set memory axis

    maxt = max(table['training_metric.memory'])
    maxp = max(table['production_metric.mean_memory'])
    axs[1].set_yticks(np.arange(0, max(maxt, maxp)+1, max(maxt, maxp)/4))
    axs[1].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[1].set_xlim([0, nrows - 1])
    # ax2 = axs[1].twinx()
    axs[1].set_ylabel('memory\n(MiB)')
    axs[1].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[1].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[1].get_yticks())
    # ax2.yaxis.set_major_locator(FixedLocator([0.5,2,8]))
    # ax2.yaxis.set_minor_locator(FixedLocator([1,4]))
    # ax2.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    # ax2.tick_params(which='major', pad=20, axis='y')

    # set throughput axis

    maxt = max(table['training_metric.throughput'])
    maxp = max(table['production_metric.mean_throughput'])
    axs[2].set_yticks(np.linspace(0, max(maxt, maxp), 5))
    axs[2].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[2].set_xlim([0, nrows - 1])
    # ax2 = axs[2].twinx()
    axs[2].set_ylabel('rqs')
    axs[2].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[2].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[2].get_yticks())
    # axs[2].set_ylim([0, 10000])

    # set process axis

    maxt = max(table['training_metric.process_time'])
    maxp = max(table['production_metric.mean_process_time'])
    axs[3].set_yticks(np.linspace(0, max(maxt, maxp), 4))
    axs[3].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[3].set_xlim([0, nrows - 1])
    # ax2 = axs[3].twinx()
    axs[3].set_ylabel('resp. time(s)')
    axs[3].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[3].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[3].get_yticks())
    # axs[2].set_ylim([0, 10000])

    # axs[2].set_ylim([0, 10000])
    # print(table['training_metric.process_time'])
    # print(table['production_metric.process_time'])

    # set errors axis
    maxt = max(table['training_metric.errors'])
    maxp = max(table['production_metric.mean_errors'])
    axs[4].set_yticks(np.linspace(0, max(maxt, maxp), 4))
    axs[4].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[4].set_xlim([0, nrows - 1])
    # axs[4].set_ylim([0, 100])
    # ax2 = axs[4].twinx()
    axs[4].set_ylabel('errors (%)')
    axs[4].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[4].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[4].get_yticks())

    # set objective axis

    maxt = max(table['training_metric.objective'])
    maxp = max(table['production_metric.min_objective'])
    axs[5].set_yticks(np.linspace(0, max(maxt, maxp), 4))
    axs[5].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[5].set_xlim([0, nrows - 1])
    # ax2 = axs[5].twinx()
    axs[5].set_ylabel('requests/$')
    axs[5].tick_params(labeltop=False, labelright=True, labelleft=True)
    axs[5].yaxis.set_ticks_position('both')
    # ax2.set_yticks(axs[5].get_yticks())
    # axs[5].set_ylim([0, max(table['training_metric.objective'])])

    axs[1].legend(frameon=False, )
    for i in range(1,nmetrics):
        axs[i].get_legend().remove()
        axs[i].margins(x=0)

    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(mlines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=10, markeredgewidth=1.5))

    axs[0].legend(handles, ['production', 'training', 'tuning'], bbox_to_anchor=(0., 1.3, 1., .102),
                  loc='upper center',
                  ncol=3, borderaxespad=0., frameon=False, )
    axs[-1].set_xlabel('iterations')
    axs[-1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0].set_title(title, loc='left')

    plt.show()

def newline_yspan(p1, p2, ax):
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin, ymax], color='black', linestyle='-', linewidth=0.3)
    ax.add_line(l)
    return l

def newline(p1, p2, ax, arrow=False, **kwargs):
    xmin, xmax = ax.get_xbound()

    # if(p2[0] == p1[0]):
    #     xmin = xmax = p1[0]
    #     ymin, ymax = ax.get_ybound()
    # else:
    # ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
    # ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    if arrow:
        if p1[1] > p2[1]:
            ax.scatter(p2[0], p2[1], marker='^', color=kwargs['color'])
        elif p2[1] > p1[1]:
            ax.scatter(p1[0], p1[1], marker='v', color=kwargs['color'])


    l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]],**kwargs)
    ax.add_line(l)
    return l

def plot_training(p1, p2, train, ax, **kwargs):

    if p1[1] < train[1] < p2[1]:
        ax.scatter(train[0], train[1], marker='x', color=kwargs['color'])
    else:
        if max(p1[1], p2[1]) > train[1]:
            ax.scatter(train[0], train[1], marker='x', color=kwargs['color'])
        else :
            ax.scatter(train[0], train[1], marker='x', color=kwargs['color'])


        l = mlines.Line2D([p1[0], p2[0]], [train[1], max(p1[1], p2[1])],**kwargs)
        ax.add_line(l)
        return l

def plot_configs(table:pd.DataFrame, title):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    colormap = cm.get_cmap('jet')

    def annotate_dots(t:pd.DataFrame, ax, idxs):
        memoization = {}
        for k, v in t.iterrows():
            if (v[idxs[0]], v[idxs[1]]) in memoization:
                annotation:Annotation = memoization[(v[idxs[0]], v[idxs[1]])]
                annotation.set_text(f'{max(float(annotation.get_text()), v[-1]):.4}')
            else:
                value = v[-1]
                memoization[(v[idxs[0]], v[idxs[1]])] = ax.annotate(f'{value:.4}', (v[idxs[0]], v[idxs[1]]),
                                                                    # xytext = (10, -5),
                                                                    # textcoords = 'offset points',
                                                                    # family = 'sans-serif',
                                                                    fontsize = 8,
                                                                    # color = 'darkslategrey'
                                                                    )


    df.plot.scatter(
        ax=axs[0][0],
        x='last_config.daytrader-service.cpu',
        y='last_config.daytrader-service.memory',
        c='production_metric.objective',
        cmap=colormap,
        colorbar=False
    )
    annotate_dots(table, axs[0][0], [0,1])

    df.plot.scatter(
        ax=axs[0][1],
        x='last_config.daytrader-config-app.CONMGR1_MAX_POOL_SIZE',
        y='last_config.daytrader-service.memory',
        c='production_metric.objective',
        cmap=colormap,
        colorbar=False
    )
    annotate_dots(table, axs[0][1], [2, 1])

    df.plot.scatter(
        ax=axs[1][0],
        x='last_config.daytrader-config-app.HTTP_MAX_KEEP_ALIVE_REQUESTS',
        y='last_config.daytrader-service.memory',
        c='production_metric.objective',
        cmap=colormap,
        colorbar=False
    )
    annotate_dots(table, axs[1][0], [3, 1])

    df.plot.scatter(
        ax=axs[1][1],
        x='last_config.daytrader-config-app.MAX_THREADS',
        y='last_config.daytrader-service.memory',
        c='production_metric.objective',
        cmap=colormap,
        colorbar=False
    )
    annotate_dots(table, axs[1][1], [4, 1])

    fig.tight_layout()
    cax, kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    plt.colorbar(axs[0][0].get_children()[0], cax=cax, **kw)

    plt.show()

def plot_box(table:pd.DataFrame, title:str):
    # make objetive positive
    table['production_metric.objective'] *= -1
    table['training_metric.objective'] *= -1
    # transform bytes to GiB
    table['production_metric.memory'] /= 2 ** 20
    table['training_metric.memory'] /= 2 ** 20
    # transform errors to %
    table['production_metric.errors'] *= 100
    table['training_metric.errors'] *= 100
    # table['tuned'] = table['tuned'].apply(lambda row: 0 if row == 'false' or row == False else 1)

    # daytrader parameters
    columns = [
        'last_config.daytrader-config-app.CONMGR1_AGED_TIMEOUT',
        'last_config.daytrader-config-app.CONMGR1_MAX_IDLE_TIMEOUT',
        'last_config.daytrader-config-app.CONMGR1_MAX_POOL_SIZE',
        'last_config.daytrader-config-app.CONMGR1_MIN_POOL_SIZE',
        'last_config.daytrader-config-app.CONMGR1_REAP_TIME',
        'last_config.daytrader-config-app.CONMGR1_TIMEOUT',
        'last_config.daytrader-config-app.HTTP_MAX_KEEP_ALIVE_REQUESTS',
        'last_config.daytrader-config-app.HTTP_PERSIST_TIMEOUT',
        'last_config.daytrader-config-app.MAX_THREADS',
        'last_config.daytrader-service.cpu',
        'last_config.daytrader-service.memory',
    ]
    nrows = len(table.index)
    new_colors = []
    hashes = []
    memoization = {}
    SEED=0

    # makes python hash reproducible
    hashseed = os.getenv('PYTHONHASHSEED')
    if not hashseed:
        os.environ['PYTHONHASHSEED'] = str(SEED)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # maps each list of parameters tuned to a hash associate to each hash a unique color
    for index, row in table.iterrows():
        unique = hashlib.md5(bytes(str(tuple([float(item) for item in row[columns].values])), 'ascii')).hexdigest()
        print(unique, row[columns].values)
        if not unique in memoization:
            memoization[unique] = index / nrows
            memoization[unique] = abs(hash(unique)) / sys.maxsize

        if unique in memoization:
            new_colors.append(memoization[unique])

        # unique = hash(tuple(row[columns].values))
        # print(unique)
        hashes.append(unique)

    # add hashes and colors to table
    table['config'] = hashes
    table['configs'] = new_colors
    columns = [
        'config_to_eval.daytrader-config-app.CONMGR1_AGED_TIMEOUT',
        'config_to_eval.daytrader-config-app.CONMGR1_MAX_IDLE_TIMEOUT',
        'config_to_eval.daytrader-config-app.CONMGR1_MAX_POOL_SIZE',
        'config_to_eval.daytrader-config-app.CONMGR1_MIN_POOL_SIZE',
        'config_to_eval.daytrader-config-app.CONMGR1_REAP_TIME',
        'config_to_eval.daytrader-config-app.CONMGR1_TIMEOUT',
        'config_to_eval.daytrader-config-app.HTTP_MAX_KEEP_ALIVE_REQUESTS',
        'config_to_eval.daytrader-config-app.HTTP_PERSIST_TIMEOUT',
        'config_to_eval.daytrader-config-app.MAX_THREADS',
        'config_to_eval.daytrader-service.cpu',
        'config_to_eval.daytrader-service.memory',
    ]
    nrows = len(table.index)
    new_colors = []
    hashes = []
    # memoization = {}
    print()
    # maps each list of parameters tuned to a hash associate to each hash a unique color
    for index, row in table.iterrows():
        unique = hashlib.md5(bytes(str(tuple([float(item) for item in row[columns].values])), 'ascii')).hexdigest()
        print(unique, row[columns].values)

        if not unique in memoization:
            memoization[unique] = index / nrows
            memoization[unique] = abs(hash(unique)) / sys.maxsize

        if unique in memoization:
            new_colors.append(memoization[unique])

        # unique = hash(tuple(row[columns].values))
        # print(unique)
        hashes.append(unique)
    # add hashes and colors to table
    table['tconfig'] = hashes
    table['tconfigs'] = new_colors
    # expands short table with NaN values
    reduced_table = table[['training_metric.objective','production_metric.objective', 'config', 'configs', 'tconfig', 'tconfigs']].reset_index()
    reduced_table['max'] = [float('nan') for _  in table.index]
    reduced_table['min'] = [float('nan') for _ in table.index]
    reduced_table['mean'] = [float('nan') for _ in table.index]

    # group values by config
    max_table = reduced_table.groupby((reduced_table.config!=reduced_table.config.shift()).cumsum()).max().reset_index(drop=True)
    min_table = reduced_table.groupby((reduced_table.config!=reduced_table.config.shift()).cumsum()).min().reset_index(drop=True)
    mean_table = reduced_table.groupby((reduced_table.config!=reduced_table.config.shift()).cumsum()).mean().reset_index(drop=True)

    # update values to larger table
    for index, row in max_table.iterrows():
        reduced_table.at[row['index'],'max'] = max_table.iloc[index]['production_metric.objective']
        reduced_table.at[row['index'],'min'] = min_table.iloc[index]['production_metric.objective']
        reduced_table.at[row['index'],'mean'] = mean_table.iloc[index]['production_metric.objective']

    # use ffill to propagate last valid observation forward to next valid, the oposite of bfill
    reduced_table = reduced_table.fillna(method='bfill')

    # use generate a color pallete
    from SecretColors.cmaps import ColorMap, TableauMap
    from SecretColors import Palette
    # cm = ColorMap(matplotlib)
    cm = TableauMap(matplotlib)
    # p = Palette('ibm', seed=SEED)
    # my_colors = [p.red(shade=30), p.white(), p.blue(shade=60)]
    # my_colors = p.random_gradient(no_of_colors=10)
    colormap = cm.colorblind()#cm.from_list(p.random(nrows, seed=SEED))

    # adjust postition at x-axis
    reduced_table['iterations'] = [i + 0.5 for i in range(len(reduced_table))]

    # plotting
    # ax = reduced_table.plot(x='iterations', drawstyle='steps-mid', y='max', color='black', linewidth=0.5)
    ax = reduced_table.plot.scatter(x='iterations', y='max', color='black', marker='_')
    # ax = reduced_table.plot(ax=ax, x='iterations', drawstyle='steps-mid', y='min', color='black', linewidth=0.5)
    ax = reduced_table.plot.scatter(ax=ax, x='iterations', y='min', color='black', marker='_')
    ax = reduced_table.plot.scatter(ax=ax, x='iterations', y='mean', color='black', marker='o')
    # ax.fill_between(reduced_table['iterations'], reduced_table['min'], reduced_table['max'], step='mid', alpha=0.7, color='black')
    ax = reduced_table.plot.scatter(ax=ax, x='iterations', y='production_metric.objective', marker='*', color='red')
    # ax = reduced_table.plot.scatter(ax=ax, x='iterations', y='training_metric.objective', marker='x', color='red')

    # split chart by configs and paint each region with a unique color
    cmap = matplotlib.cm.get_cmap(colormap)
    k = 3
    count =1
    top = max(reduced_table['max']) + 18

    for x, yp, yt, c, tc, _min, _max, mean in zip(
            reduced_table['iterations'],
            reduced_table['production_metric.objective'],
            reduced_table['training_metric.objective'],
            reduced_table['config'],
            reduced_table['tconfig'],
            reduced_table['min'],
            reduced_table['max'],
            reduced_table['mean']):
        ax.text(x, -20+2.5*((-1)**count), c[:3], {'ha': 'center', 'va': 'bottom'}, rotation=0, fontsize='smaller', color='red')
        ax.text(x, top-2.5*((-1)**count), tc[:3], {'ha': 'center', 'va': 'top'}, rotation=0, fontsize='smaller', color='blue')
        ax.axvspan(x-0.5, x+0.5, facecolor=cmap(memoization[c]), alpha=0.5)
        plot_training([x, _min], [x, _max], [x, yt], ax, color='blue', linestyle='--', linewidth=0.7)
        newline([x, _min], [x, _max], ax, color='black', linestyle='-', linewidth=0.7)

        count += 1


    # add divisions at every tuning applied
    for it, istuned in enumerate(table['tuned']):
        # print(it, table.iloc[it]['config'])
        if istuned != 'false':
            if it+1 < len(table) and table.iloc[it]['config'] != table.iloc[it+1]['config']:
                newline_yspan([it + 1, 0], [it + 1, 10], ax)
            
    # customize x-ticks
    ax.xaxis.set_ticks([])
    # Hide major tick labels
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    # Customize minor tick labels
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([i+0.5 for i in range(nrows)]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter([str(i) for i in range(nrows)]))
    ax.set_xlim([0, nrows])

    # customize legend
    handles, labels = ax.get_legend_handles_labels()
    # handles.pop()
    # handles.pop()
    # handles.append(matplotlib.patches.Patch(facecolor='black', edgecolor='k', alpha=0.7,
    #                      label='max-min region'))
    handles.append(
        mlines.Line2D([], [], color='black', marker='o', linestyle='-'))
    handles.append(
        mlines.Line2D([], [], color='red', marker='*', linestyle='None'))
    handles.append(
        mlines.Line2D([], [], color='blue', marker='x', linestyle='None'))
    handles.append(
        mlines.Line2D([], [], color='black', marker='_', linestyle='None'))

    handles.append(matplotlib.patches.Patch(facecolor=cmap(memoization[c]), edgecolor='k', alpha=0.7,
                                                                 label='config. color'))

    # handles.append(
    #     mlines.Line2D([], [], color='black', marker='|', linestyle='None'))

    ax.legend(handles, [
        'prod. avg. at config. \'abc\'',
        'prod. value at n-th iteration',
        'train. value at n-th iteration',
        'max-min',
        'config. \'abc\' color',
        # 'train. > prod.'
    ],frameon=True, bbox_to_anchor=(1, 1.08), loc='center',fontsize='small')
    ax.set_title(title, loc='left')
    ax.set_ylabel('requests/$')

    plt.text(0.1, 0.86, 'train.\nconfig.', fontsize='smaller', transform=plt.gcf().transFigure)
    plt.text(0.1, 0.11, 'prod.\nconfig.', fontsize='smaller', transform=plt.gcf().transFigure)
    plt.show()

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    # df = load_data('./resources/logging-202009281720.csv')
    # title = 'AcmeAir'
    # df = load_data('./resources/logging-202010010950.csv')
    # df = load_data('./resources/logging-202010021110.csv')
    # df = load_data('./resources/logging-202010061050.csv')
    # df = load_data('./resources/logging-202010071200.csv')
    # df = load_data('./resources/logging-202010081630.csv')
    # df = load_data('./resources/logging-202010171030.csv')
    # df = load_data('./resources/logging-202010231030.csv')

    # df = load_data('./resources/logging-trxrhel-202011201540.csv')
    # df = load_data('./resources/logging-trxrhel-202011210032.csv')

    # df = load_data('./resources/logging-trxrhel-202011221630.csv')
    # df = load_data('./resources/logging-trxrhel-202011241500.csv')

    df = load_data('./resources/logging-trxrhel-202011261015.csv')
    # df = load_data('./resources/logging-trxrhel-202012021945.csv')

    # df = load_data('./resources/logging-trinity-202011191237.csv')
    title = 'DayTrader'
    # mtable, wtable, ctable = split_table(df)

    # plot_metrics(mtable, title)
    # plot_metrics_summarized(df, title)
    # plot_configs(ctable, title)
    plot_box(df, title)

