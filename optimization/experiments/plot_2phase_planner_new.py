import hashlib
import heapq
import json
import math
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

import matplotlib
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.table import Table
from pandas.plotting import scatter_matrix

SEED = 0


def reset_seeds():
    np.random.seed(123)
    random.seed(123)
    hashseed = os.getenv('PYTHONHASHSEED')
    if not hashseed:
        os.environ['PYTHONHASHSEED'] = str(SEED)
        os.execv(sys.executable, [sys.executable] + sys.argv)


reset_seeds()


def load_raw_data(filename: str, service_name: str, workload: str, skip_reset=False, skip_pruned=False,
                  skip_tuned=True, show_workload_gap=True) -> pd.DataFrame:
    raw_data = []

    def name(data):
        return hashlib.md5(bytes(str(data.items()), 'ascii')).hexdigest()

    def nbest(trials: list[dict], nursery: list[dict], tenured: list[dict], n: int) -> dict:
        states = {trial['uid']: trial['state'] for trial in trials}

        @dataclass
        class Configuration:
            name: str
            uid: int
            value: float

            def __hash__(self):
                return hash(self.name)

            def __lt__(self, other):
                return self.value < other.value

            # def __eq__(self, other):
            #     return self.name == other.name

            def __repr__(self):
                return f'Configuration(uid={self.uid}, name={self.name},  value={self.value})'

        nursery = [Configuration(item['name'], item['uid'], item['value']) for item in nursery]
        tenured = [Configuration(item['name'], item['uid'], item['value']) for item in tenured]

        trials = {trial['uid']: trial['state'] for trial in trials}
        tmp = [cfg for cfg in set(nursery + tenured) if trials[cfg.uid] == 'COMPLETE']
        heapq.heapify(tmp)

        if len(tmp) == 0:
            return {}
        if n > 1:
            return {i: item.name[:3] for i, item in enumerate(heapq.nsmallest(n, tmp))}

    with open(filename) as jsonfile:
        for i, row in enumerate(jsonfile):
            # workaround to avoid problems with mongodb id
            row: dict
            row = re.sub(r'{"_id":{"\$oid":"[a-z0-9]+"},', '{', row)
            record = json.loads(row)

            if workload != '' and record['ctx_workload']['name'] != workload:
                if show_workload_gap:
                    raw_data.append(Iteration().__dict__)
                continue

            if skip_pruned and record['pruned']:
                # if len(raw_data) > 0:
                #     raw_data = raw_data[:-1]
                continue

            if skip_reset and record['reset']:
                continue

            if skip_tuned and 'TunedIteration' == record['status']:
                def delete_training(record):
                    for key, value in record.items():
                        new_value = None
                        if isinstance(value, dict):
                            new_value = delete_training(value)
                        else:
                            if isinstance(value, Number):
                                new_value = 0
                            elif isinstance(value, str):
                                new_value = ''
                            elif isinstance(value, list):
                                new_value = []
                            elif isinstance(value, bool):
                                new_value = False
                        record[key] = new_value
                    return record

                #
                #     # delete_training(record['training'])
                continue

            raw_data.append(Iteration(
                # pruned=record['curr_workload']['name'] != record['ctx_workload']['name'],
                pruned=record['pruned'],
                workload=record['curr_workload']['name'],
                iteration=record['global_iteration'],
                pname=record['production']['curr_config']['name'],
                pscore=math.fabs(record['production']['curr_config']['score'] or 0),
                pmean=math.fabs(record['production']['curr_config']['stats']['mean'] or 0),
                pmedian=math.fabs(record['production']['curr_config']['stats']['median'] or 0),
                pmad=math.fabs(record['production']['curr_config']['stats'].get('mad', np.nan)),
                pmin=math.fabs(record['production']['curr_config']['stats']['min'] or 0),
                pmax=math.fabs(record['production']['curr_config']['stats']['max'] or 0),
                pstddev=record['production']['curr_config']['stats']['stddev'] or 0,
                ptruput=record['production']['metric']['throughput'],
                pproctime=record['production']['metric']['process_time'],
                pmem=(record['production']['metric']['memory'] or 0) / 2 ** 20,
                pmem_lim=(record['production']['metric'].get('memory_limit',
                                                             (record['production']['curr_config']['data'][service_name][
                                                                  'memory'] or 0) * 2 ** 20) or 0) / 2 ** 20,
                preplicas=math.ceil(record['production']['metric'].get('curr_replicas', 1)),
                pparams=Param(record['production']['curr_config']['data'] or {},
                              math.fabs(record['production']['curr_config']['score'] or 0)),
                tname=record['training']['curr_config']['name'],
                tscore=math.fabs(record['training']['curr_config']['score'] or 0),
                tmean=math.fabs(record['training']['curr_config']['stats']['mean'] or 0),
                tmedian=math.fabs(record['training']['curr_config']['stats']['median'] or 0),
                tmad=math.fabs(record['training']['curr_config']['stats'].get('mad', np.nan)),
                tmin=math.fabs(record['training']['curr_config']['stats']['min'] or 0),
                tmax=math.fabs(record['training']['curr_config']['stats']['max'] or 0),
                tstddev=record['training']['curr_config']['stats']['stddev'],
                ttruput=record['training']['metric']['throughput'],
                tproctime=record['training']['metric']['process_time'],
                tmem=(record['training']['metric']['memory'] or 0) / 2 ** 20,
                tmem_lim=(record['training']['metric'].get('memory_limit',
                                                           (record['training']['curr_config']['data'][service_name][
                                                                'memory'] or 0) * 2 ** 20) or 0) / 2 ** 20,
                treplicas=math.ceil(record['training']['metric'].get('curr_replicas', 1)),
                tparams=Param(record['training']['curr_config']['data'],
                              math.fabs(record['training']['curr_config']['score'] or 0)),
                # create an auxiliary table to hold the 3 best config at every iteration
                nbest=nbest(record['trials'], record['nursery'], record['tenured'], 3)
                # nbest={chr(i + 97): best['name'][:3] for i, best in enumerate(record['best'])} if 'best' in record else {}
            ).__dict__)

    return pd.DataFrame(raw_data).reset_index()


class Param:
    """ Config params holder"""

    def __init__(self, params, score):
        self.params: dict[str:dict[str:Any]] = params
        self.params['score'] = score


@dataclass
class Iteration:
    pruned: bool = False
    workload: str = ''
    iteration: int = 0
    pname: str = ''
    pscore: float = 0
    pmean: float = 0
    pmedian: float = 0
    pmad: float = np.nan
    pmin: float = 0
    pmax: float = 0
    pstddev: float = 0
    ptruput: float = 0
    pproctime: float = 0
    pmem: int = 0
    pmem_lim: int = 0
    preplicas: int = 0
    pparams: Param = 0
    tname: str = ''
    tscore: float = 0
    tmean: float = 0
    tmedian: float = 0
    tmad: float = np.nan
    tmin: float = 0
    tmax: float = 0
    tstddev: float = 0
    ttruput: float = 0
    tproctime: float = 0
    tmem: int = 0
    tmem_lim: int = 0
    treplicas: int = 0
    tparams: Param = 0
    nbest: dict = field(default_factory=dict)


def calculate_mad(df: pd.DataFrame):
    df['pmad'] = df.groupby('pname')['pscore'].transform(lambda x: abs(x - x.rolling(len(df), 1).median()))
    df['pmad'] = df.groupby('pname')['pmad'].transform(lambda x: x.rolling(len(df), 1).median())
    df['tmad'] = df.groupby('tname')['tscore'].transform(lambda x: x.rolling(len(df), 1).median())
    df['tmad'] = df.groupby('tname')['tscore'].transform(lambda x: x.rolling(len(df), 1).median())
    return df


def plot_clean(df: pd.DataFrame, title: str, objective_label: str = '', save: bool = False, show_table: bool = False) -> \
        list[Axes]:
    pass


def plot(df: pd.DataFrame, title: str, objective_label: str = '', save: bool = False, show_table: bool = False,
         simple_visualization=False) -> list[
    Axes]:
    if df['pmad'].dropna().empty and df['tmad'].dropna().empty:
        df = calculate_mad(df)

    # pip install SecretColors
    # create a color map
    from SecretColors.cmaps import TableauMap
    cm = TableauMap(matplotlib)
    colormap = cm.colorblind()

    reduced_table = df
    memoization = {}
    new_colors = []

    def unique_color(memoization, name):
        # hold full and short config name
        memoization[name] = abs(hash(name)) / sys.maxsize
        memoization[name[:3]] = memoization[name]

    for index, row in reduced_table.iterrows():
        # create a color table for all configs
        punique = row['pname']
        tunique = row['tname']
        wunique = row['workload']

        if punique not in memoization:
            # create a unique color table
            unique_color(memoization, punique)

        if punique in memoization:
            new_colors.append(memoization[punique])

        if tunique not in memoization:
            # update color table with training configs
            unique_color(memoization, tunique)

        if tunique in memoization:
            new_colors.append(memoization[tunique])

        if wunique not in memoization:
            unique_color(memoization, wunique)

        if wunique in memoization:
            new_colors.append(memoization[wunique])

    # plotting elements
    fig: Figure
    ax: Axes  # iterations
    ax_r: Axes  # response time row
    ax_m: Axes  # memory row
    ax_t: Axes  # truput row
    # fig = plt.figure()
    fig = plt.figure(figsize=(32, 8))
    gs = fig.add_gridspec(nrows=4, hspace=0.1, height_ratios=[1, 1, 1, 4])
    axs = gs.subplots(sharex='col')
    # bottom-up rows
    ax = axs[3]  # iterations
    ax_r = axs[2]  # ...
    ax_m = axs[1]
    ax_t = axs[0]

    # split chart by configs and paint each region with a unique color
    cmap = matplotlib.cm.get_cmap(colormap)
    # find highest point in iteration row

    # complete plot
    top = max([
        reduced_table['pscore'].max(),
        reduced_table['pmedian'].max() + reduced_table['pmad'].max(),
        reduced_table['tscore'].max(),
        reduced_table['tmedian'].max() + reduced_table['tmad'].max()
    ])

    # print(top, reduced_table['pscore'].max(), reduced_table['pstddev'].max(), reduced_table['tscore'].max(), reduced_table['tstddev'].max())
    magic_number = 0
    ax.set_ylim(ymax=top + magic_number)
    for index, row in reduced_table.iterrows():
        # max min score boundaris at every iteration
        # _tmax = max(row['pmean'], row['tscore'])
        _tmax = max(row['pmedian'], row['tscore'])
        # _tmin = min(row['pmean'], row['tscore'])
        _tmin = min(row['pmedian'], row['tscore'])

        # _pmax = max(row['pmean'], row['pscore'])
        _pmax = max(row['pmedian'], row['pscore'])
        # _pmin = min(row['pmean'], row['pscore'])
        _pmin = min(row['pmedian'], row['pscore'])

        # configuration label
        ax.text(index, 0, f"{row['pname'][:3]}", {'ha': 'center', 'va': 'bottom'}, rotation=45, fontsize='x-small',
                color='red')  # production
        ax.text(index, top + magic_number, f"{row['tname'][:3]}", {'ha': 'center', 'va': 'top'}, rotation=45,
                fontsize='x-small',
                color='blue')  # training

        # draw delta(score, (max,min)) -- residual
        plot_training([index, _tmin], [index, _tmax], [index, row['tscore']], ax, color='blue', marker='x',
                      linestyle='--', linewidth=0.4)
        plot_training([index, _pmin], [index, _pmax], [index, row['pscore']], ax, color='red', marker='None',
                      linestyle='--', linewidth=0.4)

    # paint full column
    interval = ax.xaxis.get_data_interval()
    interval = np.linspace(interval[0], interval[1], len(reduced_table))
    for i, pos in enumerate(interval[:-1]):
        delta = interval[i + 1] - pos
        if i == 0:
            x0 = pos - delta
        else:
            x0 = xf
        xf = pos + delta / 2
        rectangle: Polygon = ax.axvspan(x0, xf, facecolor=cmap(memoization[reduced_table.iloc[i]['pname']]), alpha=0.5)
        if reduced_table.iloc[i]['pruned']:
            rectangle.set_hatch('///')
        # add divisions between iterations
        newline_yspan([xf, 0], [xf, top], ax)

    x0 = xf
    xf += xf
    rectangle: Polygon = ax.axvspan(x0, xf, facecolor=cmap(memoization[reduced_table.iloc[i + 1]['pname']]), alpha=0.5)
    if reduced_table.iloc[i + 1]['pruned']:
        rectangle.set_hatch('///')

    # truput row
    ax_t = reduced_table.plot.bar(ax=ax_t, x='index', y=['ptruput', 'ttruput'], rot=0,
                                  color={'ptruput': 'red', 'ttruput': 'blue'}, width=0.8, alpha=0.7)
    # response time row
    ax_r = reduced_table.plot.bar(ax=ax_r, x='index', y=['pproctime', 'tproctime'], rot=0,
                                  color={'pproctime': 'red', 'tproctime': 'blue'}, width=0.8, alpha=0.7)
    # memory row -- dark shaded to better visualize the runtime consumption
    ax_m = reduced_table.plot.bar(ax=ax_m, x='index', y=['pmem', 'tmem'], rot=0, color={'pmem': 'red', 'tmem': 'blue'},
                                  width=0.8, alpha=1)
    # memory limit row -- light shaded to better visualize the memory limits
    ax_m = reduced_table.plot.bar(ax=ax_m, x='index', y=['pmem_lim', 'tmem_lim'], rot=0,
                                  color={'pmem_lim': 'red', 'tmem_lim': 'blue'}, width=0.8, alpha=0.3)

    # statistics marks p* stands for production t* stands for training
    # ax = reduced_table.plot(ax=ax, x='index', y='pmedian', color='yellow', marker='^', markersize=3, linewidth=0)

    if simple_visualization:
        ax = reduced_table.plot(ax=ax, x='index', y='pscore', color='black', linewidth=0.3)
    else:
        ax = reduced_table.plot(ax=ax, x='index', y='pmedian', color='black', marker='o', markersize=3, yerr='pmad',
                                linewidth=0, elinewidth=0.7, capsize=3)

    #
    #
    ax = reduced_table.plot(ax=ax, x='index', y='pmedian', color='black', marker='o', markersize=3, yerr='pmad',
                            linewidth=0, elinewidth=0.7, capsize=3)

    ax = reduced_table.plot(ax=ax, x='index', y='tmedian', color='lime', marker='^', markersize=3, linewidth=0)
    ax = reduced_table.plot(ax=ax, x='index', y='pscore', marker='*', markersize=4, color='red', linewidth=0)
    ax.set_ylim(ymin=0)  # force the graph to start at y-axis=0

    # show 3 best configs table
    if not show_table:
        ax.xaxis.set_ticks(range(len(reduced_table)))
        ax.set_xlabel('index')
        ax.margins(x=0)
        ax.tick_params(axis='x', which='major', labelsize='x-small')
        ax.tick_params(axis='x', which='minor', labelsize='x-small')
    else:
        # draw table "manually"
        ax.xaxis.set_ticks([])
        ax.set_xlabel('')
        ax.margins(x=0)
        table = pd.DataFrame(reduced_table['nbest'].to_dict())
        table = table.T
        table['replicas'] = reduced_table['preplicas']
        table = table.T
        table = table.fillna(value='')
        plt_table: Table

        # change replicas line position
        tmp = table.iloc[-1]
        for i in reversed(range(len(table))):
            table.iloc[i] = table.iloc[i - 1]
        table.iloc[0] = tmp

        try:
            reshaped_table = table.to_numpy().reshape(4, -1)
            plt_table = ax.table(cellText=reshaped_table, rowLoc='center',
                                 rowLabels=['replicas', '1st', '2nd', '3rd'], colLabels=reduced_table['index'],
                                 cellLoc='center',
                                 colLoc='center', loc='bottom')
            plt_table.set_fontsize('x-small')

            for pos, cell in plt_table.get_celld().items():
                cell.fill = True
                text: str = cell.get_text().get_text()
                if pos[0] != 0 and len(text) == 3 and text not in '1st2nd3rd':
                    plt_table[pos].set_facecolor(cmap(memoization[text]))
                    cell.set_alpha(0.5)
                if pos[0] == 0:
                    plt_table[pos].set_facecolor(cmap(memoization[reduced_table['pname'].iloc[pos[1]]]))
                    # plt_table[pos].set_facecolor(cmap(memoization[reduced_table['workload'].iloc[pos[1]]]))
                    cell.set_alpha(0.5)
                cell.set_linewidth(0.3)
        except ValueError:
            print('cannot plot top 3 table')

    # customize legend
    handles, labels = ax.get_legend_handles_labels()
    handles.pop()
    handles.pop()
    handles.pop()
    handles.pop()
    handles.append(
        mlines.Line2D([], [], color='black', marker='o', markersize=4, linestyle='-', linewidth=0.7))
    # handles.append(
    #     mlines.Line2D([], [], color='yellow', marker='^', linestyle='None'))
    handles.append(
        mlines.Line2D([], [], color='lime', marker='^', linestyle='None'))
    handles.append(
        mlines.Line2D([], [], color='red', marker='*', linestyle='None'))
    handles.append(
        mlines.Line2D([], [], color='blue', marker='x', linestyle='None'))
    handles.append(
        mlines.Line2D([], [], color='black', marker='', linestyle='--', linewidth=0.7))
    handles.append(matplotlib.patches.Patch(facecolor=cmap(list(memoization.values())[0]), edgecolor='k', alpha=0.7,
                                            label='config. color'))
    handles.append(
        matplotlib.patches.Patch(facecolor=cmap(list(memoization.values())[0]), edgecolor='k', alpha=0.7, hatch='///',
                                 label='config. pruned'))
    handles.append(matplotlib.patches.Patch(facecolor='red', edgecolor='k', alpha=0.7,
                                            label='config. color'))
    handles.append(matplotlib.patches.Patch(facecolor='blue', edgecolor='k', alpha=0.7,
                                            label='config. color'))

    # customize y-axis labels
    ax.get_legend().remove()
    ax_m.get_legend().remove()
    ax_r.get_legend().remove()

    if simple_visualization:
        ax.minorticks_on()
        ax.yaxis.grid(which='both')
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())

    ax_t.set_ylabel('requests')
    ax_r.set_ylabel('resp. time (s)')
    ax_t.set_ylim(0, ax_t.get_yaxis().get_data_interval()[1])
    ax_r.set_ylim(0, 1)
    # ax.set_ylim(0, top)
    ax_t.set_yticks(np.linspace(0, ax_t.get_yaxis().get_data_interval()[1], 4))
    ax_r.set_yticks(np.linspace(0, 1, 4))
    rlabels = [f'{item:.2f}' for item in np.linspace(0, 1, 4)]
    rlabels[-1] += '>'
    ax_r.set_yticklabels(rlabels)
    ax_m.set_ylim(0, ax_t.get_yaxis().get_data_interval()[1])
    ax_m.set_yticks([0, 1024, 2048, 4096, 8192])
    ax_m.set_ylabel('memory (MB)')
    ax_t.set_title(title, loc='left')
    ax_t.axes.get_xaxis().set_visible(False)
    ax_m.axes.get_xaxis().set_visible(False)
    ax_t.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
    ax_m.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
    ax_r.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
    # guarantee that legend is above the first row -- see line 180
    ax_t.legend(handles, [
        # 'avg. of config. \'abc\' in prod',
        'median of config \'abc\' in prod',
        'median of config \'abc\' in train',
        'prod. value at n-th iteration',
        'train. value at n-th iteration',
        'residual (*:prod, X:train), $y_i - \overline{Y_i}$',
        'config. \'abc\' color',
        'config. \'abc\' pruned',
        'production',
        'training',
    ], frameon=False, ncol=5, bbox_to_anchor=(0.6, 1.52), loc='upper center', fontsize='small')

    ax.set_ylabel(objective_label)
    # customize label position, -4 and 30 are magic numbers
    ax.text(-4, 0, 'prod.\ncfg.', fontsize='smaller')
    ax.text(-4, top + magic_number, 'train.\ncfg.', fontsize='smaller')

    # hack to change hatch linewidth
    mpl.rc('hatch', color='k', linewidth=0.5)
    # tight layout to avoid waste of white space
    # gs.tight_layout(fig, pad=3.0)
    gs.tight_layout(fig)
    if save:
        fig = plt.gcf()
        fig.set_size_inches((18, 8), forward=False)
        fig.savefig(title + '.pdf', dpi=150)  # Change is over here
    else:
        plt.show()
    # plt.show()
    return axs


# def newline(p1, p2, ax, arrow=False, **kwargs):
#     xmin, xmax = ax.get_xbound()
#
#     if arrow:
#         if p1[1] > p2[1]:
#             ax.scatter(p2[0], p2[1], marker='^', color=kwargs['color'])
#         elif p2[1] > p1[1]:
#             ax.scatter(p1[0], p1[1], marker='v', color=kwargs['color'])
#
#     l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
#     ax.add_line(l)
#     return l


def newline_yspan(p1, p2, ax):
    xmin, xmax = ax.get_xbound()

    if (p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax], color='black', linestyle='-', linewidth=0.3)
    ax.add_line(l)
    return l


def plot_training(p1, p2, train, ax, marker=None, **kwargs):
    """ dray residual """
    ax.scatter(train[0], train[1], marker=marker, color=kwargs['color'])

    top_y = max([train[1], p1[1], p2[1]])
    bot_y = min([train[1], p1[1], p2[1]])

    ax.scatter(train[0], train[1], marker=marker, color=kwargs['color'])

    l = mlines.Line2D([p1[0], p2[0]], [bot_y, top_y], **kwargs)
    ax.add_line(l)
    return l


def plot_app_curves(df: pd.DataFrame, service_name, config_name, title: str = ''):
    # data = {}
    # for index, row in df.iterrows():
    #     # extract config only
    #     tmp = {}
    #     tmp = row['tparams'].params['daytrader-config-jvm']
    #     tmp.update(row['tparams'].params['daytrader-config-app'])
    #     tmp.update(row['tparams'].params['daytrader-service'])
    #     tmp['score'] = row['tparams'].params['score']
    #     # workaround to avoid str values like NaN and Inf
    #     if len(data) == 0:
    #         for k, v in tmp.items():
    #             try:
    #                 data[k] = [float(v)]
    #             except:
    #                 continue
    #     else:
    #         for k, v in tmp.items():
    #             try:
    #                 data[k].append(float(v))
    #             except:
    #                 continue
    #
    # # recreate data frame
    # df = pd.DataFrame(data)
    df = features_table(df, service_name, config_name)
    # create correlation matrix
    mat_ax = scatter_matrix(df, alpha=0.5, figsize=(6, 6), diagonal='kde')
    # resize all labels to smallest size
    for row in mat_ax:
        for cel in row:
            label = cel.get_xlabel()
            cel.set_xlabel(label, fontsize='xx-small')
            label = cel.get_ylabel()
            cel.set_ylabel(label, fontsize='xx-small')
    plt.show()


def features_table(df: pd.DataFrame, service_name, config_name, params) -> pd.DataFrame:
    data = {}
    for index, row in df.iterrows():
        # extract config only
        tmp = {}
        # print(row)
        tmp = row[params].params[f'{service_name}-{config_name}-jvm']
        tmp.update(row[params].params[f'{service_name}-{config_name}-app'])
        tmp.update(row[params].params[f'{service_name}-service'])
        tmp['score'] = row[params].params['score']
        # workaround to avoid str values like NaN and Inf
        if len(data) == 0:
            for k, v in tmp.items():
                try:
                    data[k] = [float(v)]
                except:
                    continue
        else:
            for k, v in tmp.items():
                try:
                    data[k].append(float(v))
                except:
                    continue

    # recreate data frame
    df = pd.DataFrame.from_dict(data, orient='index')

    return df


def importance(df: pd.DataFrame, service_name='', config_name='', params='tparams', name='') -> list:
    # df is table of all parameters, one parameter per column, one iteration per row
    df = features_table(df, service_name=service_name, config_name=config_name, params=params).T
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # print(X)
    # print(y)

    # print(df.info())
    # print(df.head())
    # print(df.describe())

    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    # https://www.datacamp.com/community/tutorials/xgboost-in-python

    # data_dmatrix = xgb.DMatrix(data=X, label=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_estimators=10)
    xg_reg.fit(X_train, y_train)

    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))

    return {col: score for col, score in zip(X_train.columns, xg_reg.feature_importances_)}


def plot_importance(raw_data: dict):
    # fig = plt.figure()
    # gs = fig.add_gridspec(nrows=len(raw_data), hspace=0.000, height_ratios=[1 for _ in range(len(raw_data))])
    # axs = gs.subplots(sharex='row')

    # from SecretColors.cmaps import TableauMap
    # cm = TableauMap(matplotlib)
    # colormap = cm.colorblind()
    #
    # plt.set_cmap(colormap)
    df = pd.DataFrame.from_dict(raw_data, orient='index')

    print(df.to_csv())

    # ax = df.plot.bar()
    #
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, frameon=False, ncol=5, bbox_to_anchor=(0.5, 1.15), loc='upper center', fontsize='small')
    #
    #
    #
    # plt.show()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    title = 'Daytrader'
    service_name = "daytrader-service"
    config_name = "daytrader-config-app"

    name = 'trace-new-version-2021-05-07T02 33 40_186281'
    name = 'trace-2021-05-07T14 46 06 451301'
    # name = 'trace-replicas-2021-05-08T15 48 24 363084'
    name = 'trace-replicas-2021-05-08T19 28 57 796912'
    name = 'trace-2021-05-10T14 24 45 937067'
    name = 'trace-rps-2021-05-12T04 58 13 660584'
    name = 'trace-2021-05-12T20 24 26 750340'
    name = 'trace-rps-2021-05-12T20 24 26 750340'
    name = 'trace-cpu-2021-05-14T00 31 43 190151'
    name = 'trace-rps-avg-2021-05-12T20 24 26 750340 437713'
    name = 'trace-cpu-4-2021-05-18T21 06 25 427970'
    name = 'trace-cpu-jax-2021-05-18T21 06 25 427970 074876'
    name = 'trace-2021-05-23T23 19 03 068170'
    name = 'trace-2021-05-26T04 31 10 646280'  # <-- this is a valid trace
    name = 'trace-2021-05-28T00 27 18 518096'
    name = 'trace-long-2021-05-28T00 27 18 518096'
    name = 'trace-rps-2021-05-31T23 57 57 960814'
    name = 'trace-2021-06-01T20 16 53 803845'
    name = 'trace-2021-06-02T14 17 06 775916'
    name = 'trace-2021-06-02T21 12 08 787296' # <--- this
    name = 'trace-2021-06-03T21 57 28 650968'
    name = 'trace-2021-06-04T13 47 07 701796'
    name = 'trace-2021-06-07T21 50 03 626427'
    name = 'trace-2021-06-08T18 00 28 176707'
    name = 'trace-2021-06-09T23 10 11 735500'
    name = 'trace-2021-06-11T13 52 07 453913'
    name = 'trace-2021-06-15T13 45 14 783402'
    # plot_importance(data)

    for workload in [''] + [f'workload_{i}' for i in range(0, 5)]:
        try:
            print(workload)
            # if 'workload_4' != workload:
            #     continue
            df = load_raw_data('./resources/' + name + '.json', service_name, workload,
                               skip_reset=False,
                               skip_pruned=True,
                               skip_tuned=False,
                               show_workload_gap=False)
            empty = df[(df['pname'].str.len() > 0)]
            if empty.empty:
                print(f'skiping {workload}')
                continue
            plot(df, title=title + ': ' + name + '\n' + workload,
                 objective_label=r'$(1-error)*\frac{1}{(1+resp. time)} \times \frac{requests}{N \times \$}$',
                 save=False,
                 simple_visualization=False,
                 show_table=True)
            # break
            # plot_app_curves(df, service_name, config_name)
            # plot_importance(importance(df, 'daytrader', 'config'))
        except:
            traceback.print_exc()
            continue
