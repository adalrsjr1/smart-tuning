from pprint import pprint
import re
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter


def load_file_workload(filename: str):
    """
    :param filename:
    :return: pd.DataFrame[[replicas, cpu, memory, memory utilization, throughput, process time, errors, score]]
    """

    table = {
        'workload': [],
        'replicas': [],
        'cpu': [],
        'memory': [],
        'memory utilization': [],
        'throughput': [],
        'process time': [],
        'errors': [],
        'score': [],
        'cfg': [],
        'cost': [],
        'cost_req': [],
    }

    with open(filename) as jsonfile:
        for i, row in enumerate(jsonfile):
            # workaround to avoid problems with mongodb id
            row: dict
            row = re.sub(r'{"_id":{"\$oid":"[a-z0-9]+"},', '{', row)
            row = re.sub(r'\{\"\$numberLong\":\"(?P<N>[0-9]+)\"}', '\g<N>', row)
            record = json.loads(row)

            # DON'T TOUCH!!!
            # RULES FOR FILTERING OUT PRUNED OR TRANSITION RESULTS
            if (record['mostly_workload'] and record['mostly_workload']['name'] != record['ctx_workload']['name'] or
                    record['curr_workload']['name'] != record['ctx_workload']['name']):
                continue

            if record['reset']:
                continue

            table['cfg'].append(record['production']['curr_config']['name'])
            table['workload'].append(record['curr_workload']['name'])

            table['replicas'].append((record['production']['metric']['cpu_limit'] / 2))

            # table['cpu'].append(record['production']['metric']['cpu_limit'])
            table['cpu'].append(record['production']['metric']['cpu'] / 2)
            table['memory utilization'].append(record['production']['metric']['memory'])
            table['memory'].append(record['production']['metric']['memory_limit'])
            table['throughput'].append(record['production']['metric']['throughput'])
            table['process time'].append(record['production']['metric']['process_time'])
            table['errors'].append(record['production']['metric'].get('errors', 0))

            table['score'].append(record['production']['curr_config']['stats']['median'] * -1)
            # https://azureprice.net/?region=canadaeast&cores=1,16&ram=0,4096
            table['cost'].append(table['replicas'][-1] * 0.005283024 + (table['memory'][-1] / 1024) * 0.018490583)
            table['cost_req'].append(table['cost'][-1] / table['throughput'][-1])

    df = pd.DataFrame(table)
    # print(df)
    return df


def load_file_framework(filename: str, workload: str):
    """
    :param filename:
    :return: pd.DataFrame[[replicas, cpu, memory, memory utilization, throughput, process time, errors, score]]
    """

    table = {
        'workload': [],
        'replicas': [],
        'cpu': [],
        'memory': [],
        'memory utilization': [],
        'throughput': [],
        'process time': [],
        'errors': [],
        'score': [],
        'cfg': [],
        'cost': [],
        'cost_req': []
    }
    with open(filename) as jsonfile:
        for i, row in enumerate(jsonfile):
            # workaround to avoid problems with mongodb id
            row: dict
            row = re.sub(r'{"_id":{"\$oid":"[a-z0-9]+"},', '{', row)
            row = re.sub(r'\{\"\$numberLong\":\"(?P<N>[0-9]+)\"}', '\g<N>', row)
            record = json.loads(row)

            table['cfg'].append(record['production']['curr_config']['name'])
            table['workload'].append(workload)
            table['replicas'].append(math.ceil(record['production']['metric']['cpu'] / 2))
            table['cpu'].append(record['production']['metric']['cpu'] / 2)
            table['memory utilization'].append(record['production']['metric']['memory'])
            table['memory'].append(record['production']['metric']['memory_limit'] / 2 ** 20)
            table['throughput'].append(record['production']['metric']['throughput'])
            table['process time'].append(record['production']['metric']['process_time'])
            table['errors'].append(record['production']['metric']['errors'])
            table['score'].append(record['production']['metric']['objective'] * -1)
            # table['score'].append((1 / (1 + table['process time'][-1])) * (1 / (1 + table['errors'][-1])) *
            #                       (table['throughput'][-1] / (table['cpu'][-1] + table['memory'][-1])))

            # https://azureprice.net/?region=canadaeast&cores=1,16&ram=0,4096
            table['cost'].append(table['replicas'][-1] * 0.005283024 + (table['memory'][-1] / 1024) * 0.018490583)
            table['cost_req'].append(table['cost'][-1] / table['throughput'][-1])

    df = pd.DataFrame(table)
    return df


def plot_replicas(df: pd.DataFrame, length=50, title=''):
    df = df.apply(lambda x: pd.Series(x.dropna().values))[:length]
    # df.append(df.iloc[[-1] * 5])

    fig, axes = plt.subplots(nrows=len(df.columns) // 2,
                             # figsize=(6,6),
                             ncols=1,
                             sharex='all')

    df[[name for name in df.columns if not 'workload' in name]].plot(ax=axes, subplots=True, drawstyle="steps-post",
                                                                     linewidth=1)
    custom_lines = []
    ax: Axes

    workload_labels = df[[name for name in df.columns if 'workload' in name]]
    table_labels = {}
    color = None
    for i, ax in enumerate(axes):

        line: Line2D = ax.get_lines()[0]
        ws = ['1', '2', '3']

        label = ''
        for j, x in enumerate(line.get_xdata()):
            y = line.get_ydata()[j]
            label = workload_labels.loc[j][i]
            if i < 2:
                table_labels = {
                    'workload_50': 0.1,
                    'workload_100': 0.3,
                    'workload_200': 0.5,
                }
            else:
                table_labels = {
                    'workload_5': 0.1,
                    'workload_10': 0.3,
                    'workload_50': 0.5,
                    'jsp': 0.1,
                    'jsf': 0.5,
                }
            color = line.get_color()
            try:
                ax.axvspan(j, j + 1, facecolor=color, alpha=table_labels[label])
            except:
                pass

        ax.set_xlim(0, len(df))
        ax.set_xticks(np.linspace(0, len(df), 11))
        ax.set_xticklabels([int(i) for i in np.linspace(0, 150, 11)])

        ax.get_legend().remove()
        custom_lines.append(Line2D([0], [0], lw=1, color=ax.lines[0].get_color()))
        ax.grid(b=True, axis='y', which='major', linestyle='-', alpha=0.5)
        if i == len(axes) // 2:
            ax.set_ylabel('# replicas')
        ax.minorticks_off()

    custom_lines.append(matplotlib.patches.Patch(facecolor='k', edgecolor='k', alpha=0.1,
                                                 label='config. color'))
    custom_lines.append(matplotlib.patches.Patch(facecolor='k', edgecolor='k', alpha=0.3,
                                                 label='config. color'))
    custom_lines.append(matplotlib.patches.Patch(facecolor='k', edgecolor='k', alpha=0.5,
                                                 label='config. color'))
    legend_labels = list([name for name in df.columns if not 'workload' in name]) + ['workload 1', 'workload 2',
                                                                                     'workload 3']
    axes[0].legend(custom_lines, legend_labels, frameon=False, loc='upper center', fontsize='small', ncol=3,
                   bbox_to_anchor=(0.7, 1.7), )

    axes[0].set_title(title, loc='left')

    axes[-1].set_xlabel('iterations')

    fig.tight_layout()


def plot_perf(df0: pd.DataFrame, dfN: pd.DataFrame, title=''):
    dfN = dfN / df0
    df0 = df0 / df0
    # df0 = df0.T
    # dfN = dfN.T

    metrics_lst = ['score', 'memory', 'throughput', 'proc. time', 'replicas']
    # metrics_lst = ['score']
    fig, axes = plt.subplots(nrows=len(metrics_lst),
                             # figsize=(6,6),
                             ncols=1,
                             sharex='all')

    def plotter(ax: Axes, df0, dfN, metric, ylabel='', yunit=''):
        data = pd.DataFrame(df0.loc[metric])
        data.columns = ['initial cfg']
        data['final cfg'] = pd.DataFrame(dfN.loc[metric])
        data.plot.bar(ax=ax)

        ax.set_yticks(np.linspace(0, data.max().max(), 4))
        if ax.get_xticklabels():
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize='small')

        # if metric != 'throughput':
        #     ax.set_yscale('log')
        # if metric == 'process time':
        #     ax.set_yscale('symlog', linthresh=50, linscale=.5)
        #     ax.set_yticks([10, 100, 250, data.max().max()])

        # if metric == 'score':
        #     ax.set_yscale('symlog', basey=2)

        # if metric == 'throughput':
        #     ax.set_yscale('symlog', linthresh=1000, linscale=.5)
        #     ax.set_yticks([100, 500] + list(np.linspace(1000, data.max().max(), 3)))

        # if ylabel == 'score':
        #     ylabel = r'score = $\frac{1}{(1+resp. time)} \times \frac{throughput}{CPU + memory}$'
        ax.set_ylabel(ylabel)
        ax.get_yaxis().set_major_formatter(FormatStrFormatter(f'%.2f'))

    # plotter(axes, df0, dfN, 'score', ylabel='score', yunit='')
    plotter(axes[0], df0, dfN, 'score', ylabel='score', yunit='')
    plotter(axes[1], df0, dfN, 'replicas', ylabel='replicas', yunit='')
    plotter(axes[2], df0, dfN, 'memory', ylabel='memory', yunit='MB')
    plotter(axes[3], df0, dfN, 'throughput', ylabel='throughput', yunit='rps')
    plotter(axes[4], df0, dfN, 'process time', ylabel='response\ntime', yunit='ms')
    # plotter(axes[4], df0*1000, dfN*1000, 'process time', ylabel='proc. time', yunit='ms')

    custom_lines = []
    ax: Axes

    # for i, t in enumerate(zip([axes], metrics_lst)):
    for i, t in enumerate(zip(axes, metrics_lst)):
        ax = t[0]
        metric_name = t[1]
        if metric_name == 'process time':
            metric_name = 'process\ntime'
        ax.yaxis.tick_right()
        # ax.set_ylim(0, 12)
        # ax.set_yticks(range(0,12,2))
        if i > 0:
            ax.get_legend().remove()
        ax.grid(b=True, axis='y', which='major', linestyle='-', alpha=0.5)

    # axes.set_title(title, loc='left')
    axes[0].set_title(title, loc='left')
    handles, labels = axes[0].get_legend_handles_labels()
    # axes.legend(frameon=False, loc='upper center', fontsize='small', ncol=2,
    #                    bbox_to_anchor=(0.7, 1.1))
    axes[0].legend(handles, labels, frameon=False, loc='upper center', fontsize='small', ncol=2,
                   bbox_to_anchor=(0.7, 1.5))

    fig.tight_layout()


def plot_cost(df0: pd.DataFrame, dfN: pd.DataFrame, title=''):
    dfN = dfN / df0
    df0 = df0 / df0
    # df0 = df0.T
    # dfN = dfN.T

    metrics_lst = ['cost_req', 'cost']

    fig, axes = plt.subplots(nrows=len(metrics_lst),
                             # figsize=(6,6),
                             ncols=1,
                             sharex='all')

    def plotter(ax: Axes, df0, dfN, metric, ylabel='', yunit=''):
        data = pd.DataFrame(df0.loc[metric])
        data.columns = ['initial cfg']
        data['final cfg'] = pd.DataFrame(dfN.loc[metric])
        data.plot.bar(ax=ax)

        ax.set_yticks(np.linspace(0, data.max().max(), 4))
        if ax.get_xticklabels():
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize='small')

        ax.set_ylabel(ylabel)
        ax.get_yaxis().set_major_formatter(FormatStrFormatter(f'%.2f'))

    # plotter(axes, df0, dfN, 'cost_req', ylabel='cost', yunit='')
    # plotter(axes, df0, dfN, 'cost', ylabel='cost', yunit='')
    plotter(axes[0], df0, dfN, 'cost', ylabel='cost / replica', yunit='')
    plotter(axes[1], df0, dfN, 'cost_req', ylabel='cost / request', yunit='')

    custom_lines = []
    ax: Axes

    # for i, t in enumerate(zip([axes], metrics_lst)):
    for i, t in enumerate(zip(axes, metrics_lst)):
        ax = t[0]
        metric_name = t[1]
        if metric_name == 'process time':
            metric_name = 'process\ntime'
        ax.yaxis.tick_right()
        # ax.set_ylim(0, 12)
        # ax.set_yticks(range(0,12,2))
        if i > 0:
            ax.get_legend().remove()
        ax.grid(b=True, axis='y', which='major', linestyle='-', alpha=0.5)

    # axes.set_title(title, loc='left')
    axes[0].set_title(title, loc='left')
    # handles, labels = axes.get_legend_handles_labels()
    handles, labels = axes[0].get_legend_handles_labels()
    # axes.legend(frameon=False, loc='upper center', fontsize='small', ncol=2,
    #                    bbox_to_anchor=(0.7, 1.1))
    axes[0].legend(handles, labels, frameon=False, loc='upper center', fontsize='small', ncol=2,
                   bbox_to_anchor=(0.7, 1.5))

    fig.tight_layout()


def perf_table(df0: pd.DataFrame, dfN: pd.DataFrame):
    df0 = df0.T
    dfN = dfN.T
    table = pd.DataFrame(df0['score'], columns=['score'])
    table['final'] = dfN['score']

    table['summary'] = ((table['final'] / table['score']) - 1) * 100

    for name, values in zip(table.index, table.iloc):
        score, final, summary = values
        # print(f'{name} & {summary:.2f}\% \\\\'.replace('_', '\\_'))


if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 300
    # name = 'trace-quarkus-2021-08-31T18 03 04'  # ICSE (50, 100, 200)
    name = 'trace-quarkus-2021-09-14T19 46 43' # more initial memory
    df_qhd = load_file_workload(f'resources/{name}.json')

    # name = 'trace-acmeair-2021-08-31T19 27 42'  # ICSE (50, 100, 200)
    name = 'trace-acmeair-2021-09-14T19 46 28' # more initial memory
    # name = 'trace-acmeair-2021-09-14T21 09 51' # more initial memory on trinity
    df_acme = load_file_workload(f'resources/{name}.json')

    name = 'trace-daytrader-2021-08-28T00 23 06'  # ICSE (5, 10, 50)
    df_daytrader = load_file_workload(f'resources/{name}.json')

    name = 'trace-jsp-2021-03-11T13 41 07'  # ICSE JSP
    df_d_jsp = load_file_framework(f'resources/{name}.json', 'jsp')
    name = 'trace-jsf-2021-03-10T14 01 00'  # ICSE JSF
    df_d_jsf = load_file_framework(f'resources/{name}.json', 'jsf')
    np.random.seed(0)

    replicas = {
        'QHD': df_qhd[['replicas', 'workload']],
        'AcmeAir': df_acme[['replicas', 'workload']],
        'Daytrader': df_daytrader[['replicas', 'workload']],
        # 'Daytrader JSP-JSF': df_d_jsp[['replicas', 'workload']].append(df_d_jsf[['replicas', 'workload']]).sample(frac=1).reset_index(drop=True) ,
        # 'Daytrader JSP': df_d_jsp[['replicas', 'workload']],
        # 'Daytrader JSF': df_d_jsf[['replicas', 'workload']]
    }

    # # comment out the skip tuned iterations to plot the graph correctly
    df_replicas = pd.concat([pd.DataFrame(v) for k, v in replicas.items()], axis=1)

    columns = []
    for k, v in replicas.items():
        columns.append(k)
        columns.append(f'{k}_workload')
    df_replicas.columns = columns
    # df_cost.columns = columns
    plot_replicas(df_replicas, length=150, title='# replicas over time')

    row = -1


    def filter(df, workload, row, cfg_idx=0):
        df = df.loc[(df['workload'] == workload)]
        df = pd.DataFrame(df)
        cfg = df['cfg'].iloc[cfg_idx]
        # df = df.loc[(df['cfg'] == cfg)].median()
        df['cpu'] = df.groupby('cfg')['cpu'].transform(lambda s: s.rolling(3, min_periods=1).median())
        df['memory'] = df.groupby('cfg')['memory'].transform(lambda s: s.rolling(3, min_periods=1).median())
        df['memory utilization'] = df.groupby('cfg')['memory utilization'].transform(
            lambda s: s.rolling(3, min_periods=1).median())
        df['throughput'] = df.groupby('cfg')['throughput'].transform(lambda s: s.rolling(3, min_periods=1).median())
        df['process time'] = df.groupby('cfg')['process time'].transform(lambda s: s.rolling(3, min_periods=1).median())
        df['errors'] = df.groupby('cfg')['errors'].transform(lambda s: s.rolling(3, min_periods=1).median())
        df['score'] = df.groupby('cfg')['score'].transform(lambda s: s.rolling(3, min_periods=1).median())
        df['cost'] = df.groupby('cfg')['cost'].transform(lambda s: s.rolling(3, min_periods=1).median())
        df['cost_req'] = df.groupby('cfg')['cost_req'].transform(lambda s: s.rolling(3, min_periods=1).median())

        df = df.loc[(df['cfg'] == cfg)].iloc[-1]
        # print(df[['cfg','score','cost']])
        return df.drop(['cfg', 'workload'])


    perf0 = {
        'QHD_50': filter(df_qhd, 'workload_50', row),
        'QHD_100': filter(df_qhd, 'workload_100', row),
        'QHD_200': filter(df_qhd, 'workload_200', row),
        'AcmeAir_50': filter(df_acme, 'workload_50', 0),
        'AcmeAir_100': filter(df_acme, 'workload_100', 0),
        'AcmeAir_200': filter(df_acme, 'workload_200', 0),
        'Daytrader_5': filter(df_daytrader, 'workload_5', row),
        'Daytrader_10': filter(df_daytrader, 'workload_10', row),
        'Daytrader_50': filter(df_daytrader, 'workload_50', row),
        'Daytrader JSP': filter(df_d_jsp, 'jsp', row),
        'Daytrader JSF': filter(df_d_jsf, 'jsf', row)
    }
    row = -1
    cfg_idx = -1
    perfN = {
        'QHD_50': filter(df_qhd, 'workload_50', row, cfg_idx=cfg_idx),
        'QHD_100': filter(df_qhd, 'workload_100', row, cfg_idx=cfg_idx),
        'QHD_200': filter(df_qhd, 'workload_200', row, cfg_idx=cfg_idx),
        'AcmeAir_50': filter(df_acme, 'workload_50', row, cfg_idx=cfg_idx),
        'AcmeAir_100': filter(df_acme, 'workload_100', row, cfg_idx=cfg_idx),
        'AcmeAir_200': filter(df_acme, 'workload_200', row, cfg_idx=cfg_idx),
        'Daytrader_5': filter(df_daytrader, 'workload_5', row, cfg_idx=cfg_idx),
        'Daytrader_10': filter(df_daytrader, 'workload_10', row, cfg_idx=cfg_idx),
        'Daytrader_50': filter(df_daytrader, 'workload_50', row, cfg_idx=cfg_idx),
        'Daytrader JSP': filter(df_d_jsp, 'jsp', row, cfg_idx=cfg_idx),
        'Daytrader JSF': filter(df_d_jsf, 'jsf', row, cfg_idx=cfg_idx)
    }

    df_scores = pd.concat([pd.DataFrame(v, columns=[k]) for k, v in replicas.items()], axis=1)
    plot_perf(pd.DataFrame(perf0), pd.DataFrame(perfN), title='Tuning improvement')
    plot_cost(pd.DataFrame(perf0), pd.DataFrame(perfN), title='Cost reduction')
    perf_table(pd.DataFrame(perf0), pd.DataFrame(perfN))
    plt.show()
