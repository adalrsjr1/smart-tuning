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
from scipy.optimize import curve_fit
import matplotlib
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter


def load_file_workload(filename: str, iteration_lenght_minutes=5):
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
        't_workload': [],
        't_replicas': [],
        't_cpu': [],
        't_memory': [],
        't_memory utilization': [],
        't_throughput': [],
        't_process time': [],
        't_errors': [],
        't_score': [],
        't_cfg': [],
        't_cost': [],
        't_cost_req': [],
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

            # if 'Tuned' in record['status']:
            #     continue

            table['cfg'].append(record['production']['curr_config']['name'])
            table['workload'].append(record['curr_workload']['name'])
            table['replicas'].append((record['production']['metric']['cpu_limit']))
            table['cpu'].append(record['production']['metric']['cpu'])
            table['memory utilization'].append(record['production']['metric']['memory'])
            table['memory'].append(record['production']['metric']['memory_limit'])
            table['throughput'].append(record['production']['metric']['throughput'])
            table['process time'].append(record['production']['metric']['process_time'])
            table['errors'].append(record['production']['metric'].get('errors', 0))
            table['score'].append(record['production']['curr_config']['stats']['median'] * -1)
            # https://azureprice.net/?region=canadaeast&cores=1,16&ram=0,4096
            table['cost'].append((table['replicas'][-1] * 0.005283024 + (table['memory'][-1] / 1024) * 0.018490583) * 60 / iteration_lenght_minutes)
            # table['cost'].append((table['replicas'][-1] * 0.005283024 + (table['memory'][-1] / 1024) * 0.018490583) * 60 / (table['replicas'][-1]* iteration_lenght_minutes))
            try:
                table['cost_req'].append((table['cost'][-1] * table['replicas'][-1])/(table['throughput'][-1]))
            except ZeroDivisionError:
                table['cost_req'].append(table['cost'][-1])

            table['t_cfg'].append(record['training']['curr_config']['name'])
            table['t_workload'].append(record['curr_workload']['name'])
            table['t_replicas'].append((record['training']['metric']['cpu_limit']))
            table['t_cpu'].append(record['training']['metric']['cpu'])
            table['t_memory utilization'].append(record['training']['metric']['memory'])
            table['t_memory'].append(record['training']['metric']['memory_limit'])
            table['t_throughput'].append(record['training']['metric']['throughput'])
            table['t_process time'].append(record['training']['metric']['process_time'])
            table['t_errors'].append(record['training']['metric'].get('errors', 0))
            table['t_score'].append(record['training']['curr_config']['stats']['median'] * -1)
            # https://azureprice.net/?region=canadaeast&cores=1,16&ram=0,4096
            table['t_cost'].append((table['t_replicas'][-1] * 0.005283024 + (table['t_memory'][-1] / 1024) * 0.018490583) * 60 / iteration_lenght_minutes)
            try:
                table['t_cost_req'].append(table['t_cost'][-1] / (table['t_throughput'][-1]))
            except ZeroDivisionError:
                table['t_cost_req'].append(table['t_cost'][-1])

    df = pd.DataFrame(table)
    # print(df)
    return df


def load_file_framework(filename: str, workload: str, iteration_lenght_minutes=5):
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
        't_workload': [],
        't_replicas': [],
        't_cpu': [],
        't_memory': [],
        't_memory utilization': [],
        't_throughput': [],
        't_process time': [],
        't_errors': [],
        't_score': [],
        't_cfg': [],
        't_cost': [],
        't_cost_req': [],
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
            table['replicas'].append(math.ceil(record['production']['metric']['cpu']))
            table['cpu'].append(record['production']['metric']['cpu'])
            table['memory utilization'].append(record['production']['metric']['memory'])
            table['memory'].append(record['production']['metric']['memory_limit'] / 2 ** 20)
            table['throughput'].append(record['production']['metric']['throughput'])
            table['process time'].append(record['production']['metric']['process_time'])
            table['errors'].append(record['production']['metric']['errors'])
            table['score'].append(record['production']['metric']['objective'] * -1)
            # https://azureprice.net/?region=canadaeast&cores=1,16&ram=0,4096
            table['cost'].append((table['replicas'][-1] * 0.005283024 + (table['memory'][-1] / 1024) * 0.018490583)*(table['replicas'][-1]*iteration_lenght_minutes)/60)
            table['cost_req'].append((table['cost'][-1] * table['replicas'][-1])/ table['throughput'][-1])

            table['t_cfg'].append(record['training']['curr_config']['name'])
            table['t_workload'].append(workload)
            table['t_replicas'].append(math.ceil(record['training']['metric']['cpu']))
            table['t_cpu'].append(record['training']['metric']['cpu'])
            table['t_memory utilization'].append(record['training']['metric']['memory'])
            table['t_memory'].append(record['training']['metric']['memory_limit'] / 2 ** 20)
            table['t_throughput'].append(record['training']['metric']['throughput'])
            table['t_process time'].append(record['training']['metric']['process_time'])
            table['t_errors'].append(record['training']['metric']['errors'])
            table['t_score'].append(record['training']['metric']['objective'] * -1)
            # https://azureprice.net/?region=canadaeast&cores=1,16&ram=0,4096
            table['t_cost'].append((table['t_replicas'][-1] * 0.005283024 + (table['t_memory'][-1] / 1024) * 0.018490583)*iteration_lenght_minutes/60)
            try:
                table['t_cost_req'].append(table['t_cost'][-1] / table['t_throughput'][-1])
            except ZeroDivisionError:
                table['t_cost_req'].append(table['t_cost'][-1])

    df = pd.DataFrame(table)
    return df


def plot_replicas(df: pd.DataFrame, length=50, title=''):
    df = df.apply(lambda x: pd.Series(x.dropna().values))[:length]
    # df.append(df.iloc[[-1] * 5])

    fig, axes = plt.subplots(nrows=len(df.columns) // 2,
                             # figsize=(6,6),
                             ncols=1,
                             sharex='all')
    fig.set_constrained_layout(True)

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
                    'workload_jsp': 0.1,
                    'workload_jsf': 0.5,
                }
            color = line.get_color()
            try:
                ax.axvspan(j, j + 1, facecolor=color, alpha=table_labels[label])
            except:
                pass

        # ax.set_ylim(1)
        ax.set_yticks(np.linspace(0, ax.get_ylim()[1], 4, dtype=int))
        ax.set_yticklabels(np.linspace(0, ax.get_ylim()[1], 4, dtype=int))
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
    legend_labels = list([name for name in df.columns if not 'workload' in name]) + ['workload 1 or jsp', 'workload 2',
                                                                                     'workload 3 or jsf']
    axes[0].legend(custom_lines, legend_labels, frameon=False, loc='upper center', fontsize='small', ncol=3,
                   bbox_to_anchor=(0.7, 1.8), )

    axes[0].set_title(title, x=0.1)

    axes[-1].set_xlabel('iterations')

    # fig.tight_layout()


def plot_perf(df0: pd.DataFrame, dfN: pd.DataFrame, title=''):
    tmetrics = [tmetric for tmetric in df0.index if tmetric.startswith('t_')]
    df0 = df0.drop(labels=tmetrics, axis=0)
    dfN = dfN.drop(labels=tmetrics, axis=0)

    print(df0)
    print(dfN)

    dfN = dfN / df0
    df0 = df0 / df0

    dfN.fillna(1, inplace=True)
    df0.fillna(1, inplace=True)
    dfN.replace(np.inf, 1, inplace=True)
    df0.replace(np.inf, 1, inplace=True)
    # df0 = df0.T
    # dfN = dfN.T

    print(df0)
    print(dfN)


    metrics_lst = ['score', 'memory', 'throughput', 'proc. time', 'cpu']
    # metrics_lst = ['score', 'memory', 'throughput', 'replicas']
    # metrics_lst = ['score']
    fig, axes = plt.subplots(nrows=len(metrics_lst),
                             # figsize=(6,6),
                             ncols=1,
                             sharex='all')
    # fig.set_constrained_layout(True)

    def plotter(ax: Axes, df0, dfN, metric, ylabel='', yunit=''):
        data = pd.DataFrame(df0.loc[metric])
        data.columns = ['initial cfg']
        data['final cfg'] = pd.DataFrame(dfN.loc[metric])
        data.plot.bar(ax=ax)

        ax.set_yticks(np.linspace(0, data.max().max(), 4))
        # if ax.get_xticklabels():
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
    plotter(axes[1], df0, dfN, 'cpu', ylabel='replicas', yunit='')
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
        # ax.set_yticks([0, 1, ax.get_ylim()[1]])
        # ax.set_yticklabels([0, 1, ax.get_ylim()[1]])
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

    # fig.tight_layout()


def plot_cost(df0: pd.DataFrame, dfN: pd.DataFrame, title=''):
    tmetrics = [tmetric for tmetric in df0.index if tmetric.startswith('t_')]
    df0 = df0.drop(labels=tmetrics, axis=0)
    dfN = dfN.drop(labels=tmetrics, axis=0)
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
    plotter(axes[1], df0, dfN, 'cost_req', ylabel='cost / rps', yunit='')

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
    print(table)

    for name, values in zip(table.index, table.iloc):
        score, final, summary = values
        print(f'{name} & {summary:.2f}\% \\\\'.replace('_', '\\_'))


def replicas(df_acme: pd.DataFrame, df_daytrader: pd.DataFrame, df_qhd: pd.DataFrame, df_frameworks: pd.DataFrame, metric: str, title: str):
    replicas = {
        'QHD': df_qhd[[metric, 'workload']],
        'AcmeAir': df_acme[[metric, 'workload']],
        'Daytrader': df_daytrader[[metric, 'workload']],
        'Daytrader JSP-JSF': df_frameworks[[metric, 'workload']]
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
    plot_replicas(df_replicas, length=100, title=title)


def cost(df_acme: pd.DataFrame, df_daytrader: pd.DataFrame, df_qhd: pd.DataFrame, df_frameworks: pd.DataFrame):
    row = -1

    def unique_cfgs(df: pd.DataFrame, app):
        groups: pd.core.groupby.generic.DataFrameGroupBy = df[['cfg','workload']].groupby('workload')
        for group in groups:
            print(app, group[0], len(pd.DataFrame(group[1])['cfg'].unique()))

    # print(df_qhd[['cfg','workload']].groupby('workload').groups.unique())
    # print(df_daytrader[['cfg','workload']].groupby('workload').unique())
    # print(df_frameworks[['cfg','workload']].groupby('workload').unique())
    # print(df_acme[['cfg','workload']].groupby('workload').unique())

    unique_cfgs(df_acme, 'acmeair')
    unique_cfgs(df_qhd, 'qhd')
    unique_cfgs(df_daytrader, 'daytrader')
    unique_cfgs(df_frameworks, 'daytrader fw')
    def filter(df, workload, row, cfg_idx=0):
        df = df.loc[(df['workload'] == workload)]
        df = pd.DataFrame(df)
        cfg = df['cfg'].iloc[cfg_idx]
        # df = df.loc[(df['cfg'] == cfg)].median()
        df['replicas'] = df.groupby('cfg')['replicas'].transform(lambda s: s.rolling(3, min_periods=1).median())
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
        df: pd.Series
        return df.drop(['cfg', 'workload'] + [t_metric for t_metric in df.index if t_metric.startswith('t_')])

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
        'Daytrader JSP': filter(df_frameworks, 'workload_jsp', row),
        'Daytrader JSF': filter(df_frameworks, 'workload_jsf', row)
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
        'Daytrader JSP': filter(df_frameworks, 'workload_jsp', row),
        'Daytrader JSF': filter(df_frameworks, 'workload_jsf', row)
    }

    plot_cost(pd.DataFrame(perf0), pd.DataFrame(perfN), title='Cost reduction')


def cost2(df_acme: pd.DataFrame, df_daytrader: pd.DataFrame, df_qhd: pd.DataFrame, df_frameworks: pd.DataFrame, cost_per_replica=True):

    for _i, _df in enumerate([df_acme,  df_qhd, df_daytrader, df_frameworks]):
        grouped: pd.core.groupby.generic.DataFrameGroupBy = _df[['workload', 'cost', 't_cost', 'cost_req', 't_cost_req']].groupby('workload')
        fig, axes = plt.subplots(nrows=3 if _i < 3 else 2,
                                 # figsize=(6,6),
                                 ncols=1,
                                 sharex='all')

        it_lenght = {
            'acmeair': 10,
            'daytrader': 20,
            'qhd': 5,
            'daytrader jsp_jsf': 20
        }

        if _i < 2:
            workloads = {
                'workload_50': axes[0],
                'workload_100': axes[1],
                'workload_200': axes[2],
            }
        elif _i == 2:
            workloads = {
                'workload_5': axes[0],
                'workload_10': axes[1],
                'workload_50': axes[2],
            }
        else:
            workloads = {
                'workload_jsp': axes[0],
                'workload_jsf': axes[1]
            }

        app = ['AcmeAir', 'QHD', 'Daytrader', 'Daytrader JSP_JSF']

        def objective(x, a, b):
            return a*x + b

        def objective_prod(x, a, b):
            return a * np.log(x) + b

        xlen = 10000
        for i, group in enumerate(grouped):

            ax: Axes
            ax = workloads[group[0]]
            df: pd.DataFrame = group[1]
            df = df.reset_index()
            xlen = min(len(df), xlen)
            if cost_per_replica:
                df['production']  = df['cost'].cumsum()
                df['training'] = df['t_cost'].cumsum()
                df['no tuning'] = df['production'][:10]

                # df['tuned'] = pd.DataFrame([df['production'].iloc[0]])
                # df['tuned'].iloc[-1] = df['production'].iloc[-1]
                # df['tuned'] = df['tuned'].interpolate()

                popt, _ = curve_fit(objective, np.arange(10), df['no tuning'][:10])
                a, b = popt
                for row in range(10, len(df)):
                    df['no tuning'].iloc[row] = objective(df.index[row], a, b)




                # popt, _ = curve_fit(objective, np.arange(len(payoff)), payoff)
                popt, _ = curve_fit(objective_prod, np.arange(len(df))[-10:], df['production'][-10:])
                ap, bp = popt

                extra_production = df['production'].tolist()
                extra_no_tuning = df['no tuning'].tolist()
                payoff = (df['production'] - df['no tuning'] + df['training']).tolist()
                df = df.drop(columns='no tuning')
                df = df.drop(columns='production')
                l = len(df)
                row = 0
                while True:
                    value = objective_prod(l+row, ap, bp)
                    extra_production.append(value)
                    no_tuning_value = objective(l+row, a, b)
                    extra_no_tuning.append(no_tuning_value)
                    payoff_value = value - no_tuning_value
                    payoff.append(payoff_value)
                    row += 1
                    # if value - no_tuning_value > 0 or row > 1000:
                    # if row > 200:
                    if payoff_value <= 0:
                        break

                df = df.join(pd.Series(extra_production).rename('production'), how='right').reindex()
                df = df.join(pd.Series(extra_no_tuning).rename('no tuning'), how='right').reindex()

                # df = df.fillna(0)
                # payoff = []
                # for value in df['production'] - df['no tuning'] + df['training']:
                #     payoff.append(value)

                # row = 0
                # l= len(df)
                # while True:
                #     value =objective(l+row, a, b)
                #     row += 1
                #     payoff.append(value)
                #     if value <= 0 or row > 100:
                #         break
                #
                df = df.join(pd.Series(payoff).rename('payoff'), how='right')
                # print(df)



                # popt, _ = curve_fit(objective, np.arange(10), df['no tuning'][-10:])
                # a, b = popt
                # for row in range(0, len(df)-10):
                #     df['tuned'].iloc[row] = objective(df.index[row], a, b)


                ax = df[['production', 'training', 'no tuning', 'payoff']].plot(ax=ax, linewidth=1.5)
                # ax = df[['production', 'training', 'no tuning', 'tuned']].plot(ax=ax, linewidth=1.5)
            else:
                df['production']  = df['cost_req']
                df['training'] = df['t_cost_req']
                # df['cumulative loss'] = (df['t_cost_req'] - df['cost_req']).cumsum()
                # ax = df[['production', 'training', 'cumulative loss']].plot(ax=ax, drawstyle="steps-post", linewidth=1)
                ax = df[['production', 'training', 'payoff']].plot(ax=ax, drawstyle="steps-post", linewidth=1)

            ax2: Axes
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

            # ax2.set_ylabel(ax.get_ylabel())
            ax.yaxis.tick_right()
            ax2.set_yticks([])
            ax2.set_yticklabels([])

            ax.set_ylabel(group[0])
            ax.set_ylim(0)
            # ax.set_xlim(0, 40)
            # ax.set_xticks(np.linspace(0, 40, 6))
            # # ax.set_xticklabels(np.linspace(0, 50 * it_lenght[app[_i].lower()]/60, 6))
            # ax.set_xticklabels(np.linspace(0, 50, 6))
            # ax.minorticks_off()
            # ax.set_ylim(0, 10)
            # ax.set_yticks(np.linspace(0, 10, 5))
            # ax.set_yticklabels(np.linspace(0, 10, 5))
            ax.grid(b=True, axis='y', which='major', linestyle='-', alpha=0.5)



        axes[-1].set_xlabel('iterations')
        if cost_per_replica:
            axes[1].set_ylabel('cumulative cost ($)\n' + axes[1].get_ylabel())
        else:
            axes[1].set_ylabel('cost per rps ($)\n' + axes[1].get_ylabel())
        axes[0].set_title(app[_i], loc='left')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, frameon=False, loc='upper center', fontsize='small', ncol=5,
                       bbox_to_anchor=(0.7, 1.3))
        axes[1].get_legend().remove()
        if len(axes) > 2:
            axes[2].get_legend()
            axes[2].get_legend().remove()



def improvement(df_acme: pd.DataFrame, df_daytrader: pd.DataFrame, df_qhd: pd.DataFrame, df_frameworks: pd.DataFrame):
    row = -1

    def filter(df, workload, row, cfg_idx=0):
        rolling = 10

        df = df.loc[(df['workload'] == workload)]
        df = pd.DataFrame(df)
        cfg = df['cfg'].iloc[cfg_idx]
        # df = df.loc[(df['cfg'] == cfg)].median()
        df['cpu'] = df.groupby('cfg')['cpu'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        df['memory'] = df.groupby('cfg')['memory'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        df['memory utilization'] = df.groupby('cfg')['memory utilization'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        df['throughput'] = df.groupby('cfg')['throughput'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        df['process time'] = df.groupby('cfg')['process time'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        df['errors'] = df.groupby('cfg')['errors'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        df['score'] = df.groupby('cfg')['score'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        df['cost'] = df.groupby('cfg')['cost'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        df['cost_req'] = df.groupby('cfg')['cost_req'].transform(lambda s: s.rolling(rolling, min_periods=1).mean())

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
        'Daytrader JSP': filter(df_frameworks, 'workload_jsp', row),
        'Daytrader JSF': filter(df_frameworks, 'workload_jsf', row)
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
        'Daytrader JSP': filter(df_frameworks, 'workload_jsp', row, cfg_idx=cfg_idx),
        'Daytrader JSF': filter(df_frameworks, 'workload_jsf', row, cfg_idx=cfg_idx)
    }

    plot_perf(pd.DataFrame(perf0), pd.DataFrame(perfN), title='Tuning improvement')
    perf_table(pd.DataFrame(perf0), pd.DataFrame(perfN))


if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 300
    # name = 'trace-quarkus-2021-08-31T18 03 04'  # ICSE (50, 100, 200)
    name = 'trace-quarkus-2021-09-14T19 46 43'  # more initial memory
    df_qhd = load_file_workload(f'resources/{name}.json',iteration_lenght_minutes=5)

    # name = 'trace-acmeair-2021-08-31T19 27 42'  # ICSE (50, 100, 200)
    name = 'trace-acmeair-2021-09-14T19 46 28'  # more initial memory
    df_acme = load_file_workload(f'resources/{name}.json',iteration_lenght_minutes=10)
    # name = 'trace-daytrader-2021-08-28T00 23 06'  # ICSE (5, 10, 50)
    name = 'trace-daytrader-2021-09-15T23 26 02'
    name = 'trace-daytrader-2022-08-10T13_30_11'
    df_daytrader = load_file_workload(f'resources/{name}.json',iteration_lenght_minutes=20)

    # name = 'trace-daytrader-2021-09-18T00 12 14' # jsp jsf
    # name = 'trace-daytrader-2021-09-20T14 33 12' # full runing jsp much worse than expected
    # name = 'trace-daytrader-2021-09-19T14 43 41' # no tuning at all
    name = 'trace-daytrader-2021-09-22T02 42 28'

    # seip
    #name = 'trace-daytrader-2022-08-10T13_30_11'


    df_frameworks = load_file_workload(f'resources/{name}.json',iteration_lenght_minutes=20)
    # name = 'trace-jsp-2021-03-11T13 41 07'  # ICSE JSP
    # df_d_jsp = load_file_framework(f'resources/{name}.json', 'jsp',iteration_lenght_minutes=20)
    # name = 'trace-jsf-2021-03-10T14 01 00'  # ICSE JSF
    # df_d_jsf = load_file_framework(f'resources/{name}.json', 'jsf',iteration_lenght_minutes=20)
    np.random.seed(0)

    #cost2(df_acme, df_daytrader, df_qhd, df_frameworks, cost_per_replica=True)
    #replicas(df_acme, df_daytrader, df_qhd, df_frameworks, 'replicas', '# replicas over time')
    #cost(df_acme, df_daytrader, df_qhd, df_frameworks)
    improvement(df_acme, df_daytrader, df_qhd, df_frameworks)

    ### Skip iterations after tuning has ended

    plt.show()
