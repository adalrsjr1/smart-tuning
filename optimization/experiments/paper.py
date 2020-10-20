import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FixedLocator)

import matplotlib.lines as mlines
from matplotlib.text import Annotation



def load_data(filename:str) -> pd.DataFrame:
    return pd.read_csv(filename)

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
    configs_table = table[[
        'last_config.daytrader-service.cpu',
        'last_config.daytrader-service.memory',
        'last_config.daytrader-config-app.CONMGR1_MAX_POOL_SIZE',
        'last_config.daytrader-config-app.HTTP_MAX_KEEP_ALIVE_REQUESTS',
        'last_config.daytrader-config-app.MAX_THREADS',
        'production_metric.objective',
    ]]

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
        'production_metric.objective']].plot(ax=target, linewidth=0.7, drawstyle='steps-post', style=['b-']*nmetrics, subplots=True)

    table[[
        'training_metric.cpu',
        'training_metric.memory',
        'training_metric.throughput',
        'training_metric.process_time',
        'training_metric.errors',
        'training_metric.objective']].plot(ax=target, linewidth=0.7, drawstyle='steps-post', style=['r-']*nmetrics, subplots=True)

    # table[['tuned']*nmetrics].plot(ax=target, linewidth=0.7, style=['k:']*nmetrics, subplots=True, )

    for it, istuned in enumerate(table['tuned']):
        if istuned:
            newline([it, 0], [it, 10], axs[0])
            newline([it, 0], [it, 10], axs[1])
            newline([it, 0], [it, 10], axs[2])
            newline([it, 0], [it, 10], axs[3])
            newline([it, 0], [it, 10], axs[4])
            newline([it, 0], [it, 10], axs[5])

    # set cpu axis
    ax2 = axs[0].twinx()
    axs[0].set_ylabel('cpu\n(cores)')
    ax2.set_yticks([2,4,8,16])
    axs[0].set_yticks([])
    axs[0].xaxis.set_ticks([])
    axs[0].set_xlim([0, nrows-1])
    # axs[0].set_ylim([0, 16])


    # set memory axis
    ax2 = axs[1].twinx()
    axs[1].set_ylabel('memory\n(GiB)')
    ax2.set_yticks([0.5, 1, 2, 4, 8])
    axs[1].set_yticks([])
    axs[1].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[1].set_xlim([0, nrows - 1])
    # axs[1].set_ylim([0, 8])


    ax2.yaxis.set_major_locator(FixedLocator([0.5,2,8]))
    ax2.yaxis.set_minor_locator(FixedLocator([1,4]))
    ax2.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax2.tick_params(which='major', pad=20, axis='y')

    # set throughput axis
    ax2 = axs[2].twinx()
    axs[2].set_ylabel('throughput\n(rqps)')
    ax2.set_yticks(np.arange(0, max(table['training_metric.throughput'])+1, max(table['training_metric.throughput'])/4))
    axs[2].set_yticks([])
    axs[2].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[2].set_xlim([0, nrows - 1])
    # axs[2].set_ylim([0, 10000])
    print(table['training_metric.throughput'])

    # set process axis
    ax2 = axs[3].twinx()
    axs[3].set_ylabel('process time\n(s)')
    ax2.set_yticks(np.linspace(0, max(table['training_metric.process_time'])+0.1, 4))
    axs[3].set_yticks([])
    axs[3].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[3].set_xlim([0, nrows - 1])
    ax2.set_ylim([0, max(table['training_metric.process_time'])+0.1])
    # print(table['training_metric.process_time'])
    # print(table['production_metric.process_time'])

    # set errors axis
    ax2 = axs[4].twinx()
    axs[4].set_ylabel('errors (%)')
    ax2.set_yticks(np.arange(0, 125, 25))
    axs[4].set_yticks([])
    axs[4].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[4].set_xlim([0, nrows - 1])
    # axs[4].set_ylim([0, 100])

    # set objective axis
    ax2 = axs[5].twinx()
    axs[5].set_ylabel('requests/$')
    ax2.set_yticks(np.linspace(0, max(table['training_metric.objective']),4))
    axs[5].set_yticks([])
    axs[5].xaxis.set_ticks(np.arange(0, nrows, 2))
    axs[5].set_xlim([0, nrows - 1])
    # axs[5].set_ylim([0, max(table['training_metric.objective'])])

    axs[1].legend(frameon=False, )
    for i in range(1,nmetrics):
        axs[i].get_legend().remove()
        axs[i].margins(x=0)

    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(mlines.Line2D([0], [0], color='black', linestyle=':', linewidth=1))

    axs[0].legend(handles, ['production', 'training', 'tuning'], bbox_to_anchor=(0., 1.3, 1., .102),
                  loc='upper center',
                  ncol=3, borderaxespad=0., frameon=False, )
    axs[-1].set_xlabel('iterations')
    axs[-1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0].set_title(title, loc='left')

    plt.show()

def newline(p1, p2, ax):
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin, ymax], color='black', linestyle=':', linewidth=0.7)
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
    df = load_data('./resources/logging-202010171030.csv')
    title = 'DayTrader'
    mtable, wtable, ctable = split_table(df)

    # plot_metrics(mtable, title)
    plot_configs(ctable, title)

