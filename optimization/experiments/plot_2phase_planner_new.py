import hashlib
import heapq
import json
import math
import os
import pprint
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
import bson
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
                  skip_tuned=True, show_workload_gap=True, to_gib=False) -> pd.DataFrame:
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
                value = self.value or 0
                other = other.value or 0
                return value < other

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
            row = re.sub(r'\{\"\$numberLong\":\"(?P<N>[0-9]+)\"}', '\g<N>', row)
            record = json.loads(row)

            # if workload != '' and record['ctx_workload']['name'] != workload:
            if workload != '' and record['curr_workload']['name'] != workload:
                if show_workload_gap:
                    raw_data.append(Iteration().__dict__)
                continue

            # if skip_pruned and record['pruned']:
            if skip_pruned and (record['mostly_workload'] and record['mostly_workload']['name'] != record['ctx_workload']['name'] or
                record['curr_workload']['name'] != record['ctx_workload']['name']):
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

            mem_scale = 1
            if to_gib:
                mem_scale = 2 ** 20

            try:
                raw_data.append(Iteration(
                    # pruned=record['mostly_workload']['name'] != record['ctx_workload']['name'] or
                    #        record['curr_workload']['name'] != record['ctx_workload']['name'],
                    pruned=record['pruned'],
                    workload=record['ctx_workload']['name'],
                    iteration=record['global_iteration'],
                    pname=record['production']['curr_config']['name'],
                    # pname=hashlib.md5(bytes(str(record['production']['curr_config']['data']), 'ascii')).hexdigest(),
                    pscore=math.fabs(record['production']['metric']['objective'] or 0),
                    pmean=math.fabs(record['production']['curr_config']['stats']['mean'] or 0),
                    pmedian=math.fabs(record['production']['curr_config']['stats']['median'] or 0),
                    pmad=math.fabs(record['production']['curr_config']['stats'].get('mad', np.nan) or 0),
                    pmin=math.fabs(record['production']['curr_config']['stats']['min'] or 0),
                    pmax=math.fabs(record['production']['curr_config']['stats']['max'] or 0),
                    pstddev=record['production']['curr_config']['stats']['stddev'] or 0,
                    ptruput=record['production']['metric']['throughput'],
                    pproctime=record['production']['metric']['process_time'],
                    pmem=(record['production']['metric']['memory'] or 0) / mem_scale,
                    pmem_lim=(record['production']['metric'].get('memory_limit',
                                                                 (record['production']['curr_config']['data'][service_name][
                                                                      'memory'] or 0) * mem_scale) or 0) / mem_scale,
                    pcpu=(record['production']['metric']['cpu'] or 1),
                    pcpu_lim=(record['production']['metric']['cpu_limit'] or 1),
                    preplicas=math.ceil(record['production']['metric'].get('curr_replicas', 1)),
                    pparams=Param(record['production']['curr_config']['data'] or {},
                                  math.fabs(record['production']['curr_config']['score'] or 0)),
                    # tname=hashlib.md5(bytes(str(record['training']['curr_config']['data']), 'ascii')).hexdigest(),
                    tname=record['training']['curr_config']['name'],
                    tscore=math.fabs(record['training']['metric']['objective'] or 0),
                    tmean=math.fabs(record['training']['curr_config']['stats']['mean'] or 0),
                    tmedian=math.fabs(record['training']['curr_config']['stats']['median'] or 0),
                    tmad=math.fabs(record['training']['curr_config']['stats'].get('mad', np.nan) or 0),
                    tmin=math.fabs(record['training']['curr_config']['stats']['min'] or 0),
                    tmax=math.fabs(record['training']['curr_config']['stats']['max'] or 0),
                    tstddev=record['training']['curr_config']['stats']['stddev'],
                    ttruput=record['training']['metric']['throughput'],
                    tproctime=record['training']['metric']['process_time'],
                    tmem=(record['training']['metric']['memory'] or 0) / mem_scale,
                    tmem_lim=(record['training']['metric'].get('memory_limit',
                                                               (record['training']['curr_config']['data'][service_name][
                                                                    'memory'] or 0) * mem_scale) or 0) / mem_scale,
                    tcpu=(record['training']['metric']['cpu'] or 1),
                    tcpu_lim=(record['training']['metric']['cpu_limit'] or 1),
                    treplicas=math.ceil(record['training']['metric'].get('curr_replicas', 1)),
                    tparams=Param(record['training']['curr_config']['data'],
                                  math.fabs(record['training']['curr_config']['score'] or 0)),
                    # create an auxiliary table to hold the 3 best config at every iteration
                    nbest=nbest(record['trials'] or [], record['nursery'] or [], record['tenured'] or [], 3)
                    # nbest={chr(i + 97): best['name'][:3] for i, best in enumerate(record['best'])} if 'best' in record else {}
                ).__dict__)
            except Exception as e:
                print(i)
                raise e
                pprint.pprint(record)

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
    pcpu: int = 0
    pcpu_lim: int = 0
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
    tcpu: int = 0
    tcpu_lim: int = 0
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
    reduced_table['ptruput_lim'] = reduced_table['ptruput']
    reduced_table['ttruput_lim'] = reduced_table['ttruput']
    reduced_table['pproctime'] = reduced_table['pproctime'].apply(lambda x: 1 if x >= 100000 else x)
    reduced_table['tproctime'] = reduced_table['tproctime'].apply(lambda x: 1 if x >= 100000 else x)

    reduced_table['ptruput'] = reduced_table['ptruput'] * reduced_table['pproctime']
    reduced_table['ttruput'] = reduced_table['ttruput'] * reduced_table['tproctime']

    reduced_table['psvcutil'] = (reduced_table['ptruput_lim'] / (1/reduced_table['pproctime'])).fillna(0)
    reduced_table['tsvcutil'] = (reduced_table['ttruput_lim'] / (1/reduced_table['tproctime'])).fillna(0)

    # print(reduced_table[["ptruput", "ttruput"]])
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
    gs = fig.add_gridspec(nrows=6, hspace=.001, height_ratios=[1.4, 1.4, 1.4, 1.4, 1.4, 3])
    axs = gs.subplots(sharex='col')
    # bottom-up rows
    ax = axs[5]  # iterations
    ax_u = axs[4]
    ax_r = axs[3]  # ...
    ax_m = axs[2]
    ax_c = axs[1]
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
    xf = 0
    i = 0
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

    # print(reduced_table[['pscore','tscore']])

    # truput row
    # ax_t = reduced_table.plot.bar(ax=ax_t, x='index', y=['ptruput', 'ttruput'], rot=0,
    #                               color={'ptruput': 'red', 'ttruput': 'blue'}, width=0.8, alpha=1)

    ax_t = reduced_table.plot.bar(ax=ax_t, x='index', y=['ptruput_lim', 'ttruput_lim'], rot=0,
                                  color={'ptruput_lim': 'red', 'ttruput_lim': 'blue'}, width=0.8, alpha=1)
    #psvcutil
    ax_u = reduced_table.plot.bar(ax=ax_u, x='index', y=['psvcutil', 'tsvcutil'], rot=0,
                                  color={'psvcutil': 'red', 'tsvcutil': 'blue'}, width=0.8, alpha=1)


    # trending line
    # trend_values = reduced_table['psvcutil'].values.tolist()
    # # simulates more iterations after end of tuning
    # trend_values = np.array(trend_values + trend_values[-4:-1]*20)
    #
    # # cap extremely high values for better visualization
    # trend_values = np.where(trend_values > 2, 2, trend_values)
    # z = np.polyfit(range(len(trend_values)), trend_values, 1)
    # # normalize 0-1 yields very small values because the huge outliers
    # # z = np.polyfit(range(len(trend_values)), (trend_values-min(trend_values))/(max(trend_values)-min(trend_values)), 1)
    # p = np.poly1d(z)
    # ax_u.plot(reduced_table.index, p(reduced_table.index), "c--", linewidth=1)

# response time row
#     print(reduced_table[[ 'tproctime', 'pproctime']])
    ax_r = reduced_table.plot.bar(ax=ax_r, x='index', y=[ 'pproctime', 'tproctime'], rot=0,
                                  color={'pproctime': 'red', 'tproctime': 'blue'}, width=0.8, alpha=1)
    # # memory row -- dark shaded to better visualize the runtime consumption
    ax_m = reduced_table.plot.bar(ax=ax_m, x='index', y=['pmem', 'tmem'], rot=0, color={'pmem': 'red', 'tmem': 'blue'},
                                  width=0.8, alpha=1)
    # memory limit row -- light shaded to better visualize the memory limits
    ax_m = reduced_table.plot.bar(ax=ax_m, x='index', y=['pmem_lim', 'tmem_lim'], rot=0,
                                  color={'pmem_lim': 'red', 'tmem_lim': 'blue'}, width=0.8, alpha=0.3)

    # reduced_table['pmem_util'] = reduced_table['pmem'] / reduced_table['pmem_lim']
    # reduced_table['tmem_util'] = reduced_table['tmem'] / reduced_table['tmem_lim']
    # ax_m = reduced_table.plot.bar(ax=ax_m, x='index', y=['pmem_util', 'tmem_util'], rot=0,
    #                               color={'pmem_util': 'red', 'tmem_util': 'blue'},
    #                               width=0.8, alpha=1)


    # cpu row -- dark shaded to better visualize the runtime consumption
    # reduced_table['pcpu_util'] = reduced_table['pcpu'] / reduced_table['pcpu_lim']
    # reduced_table['tcpu_util'] = reduced_table['tcpu'] / reduced_table['tcpu_lim']
    # ax_c = reduced_table.plot.bar(ax=ax_c, x='index', y=['pcpu_util', 'tcpu_util'], rot=0, color={'pcpu_util': 'red', 'tcpu_util': 'blue'},
    #                               width=0.8, alpha=1)

    ax_c = reduced_table.plot.bar(ax=ax_c, x='index', y=['pcpu', 'tcpu'], rot=0, color={'pcpu': 'red', 'tcpu': 'blue'},
                                  width=0.8, alpha=1)
    ax_c = reduced_table.plot.bar(ax=ax_c, x='index', y=['pcpu_lim', 'tcpu_lim'], rot=0,
                                  color={'pcpu_lim': 'red', 'tcpu_lim': 'blue'}, width=0.8, alpha=0.3)


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
        # table = pd.DataFrame(reduced_table['nbest'].to_dict())
        # table = table.T
        # table['replicas'] = reduced_table['preplicas']
        table = reduced_table['preplicas']
        table = table.T
        table = table.fillna(value='')
        plt_table: Table

        # change replicas line position
        tmp = table.iloc[-1]
        for i in reversed(range(len(table))):
            table.iloc[i] = table.iloc[i - 1]
        table.iloc[0] = tmp

        try:
            reshaped_table = table.to_numpy().reshape(1, -1)
            plt_table = ax.table(cellText=reshaped_table, rowLoc='center',
                                 rowLabels=['replicas'], colLabels=reduced_table['index'],
                                 cellLoc='center',
                                 colLoc='center', loc='bottom')
            plt_table.set_fontsize('x-small')

            # reshaped_table = table.to_numpy().reshape(4, -1)
            # plt_table = ax.table(cellText=reshaped_table, rowLoc='center',
            #                      rowLabels=['replicas', '1st', '2nd', '3rd'], colLabels=reduced_table['index'],
            #                      cellLoc='center',
            #                      colLoc='center', loc='bottom')
            # plt_table.set_fontsize('x-small')
        #
        #     for pos, cell in plt_table.get_celld().items():
        #         cell.fill = True
        #         text: str = cell.get_text().get_text()
        #         if pos[0] != 0 and len(text) == 3 and text not in '1st2nd3rd':
        #             try:
        #                 plt_table[pos].set_facecolor(cmap(memoization[text]))
        #                 cell.set_alpha(0.5)
        #             except KeyError:
        #                 print(f'KeyError: {text}')
        #         if pos[0] == 0:
        #             plt_table[pos].set_facecolor(cmap(memoization[reduced_table['pname'].iloc[pos[1]]]))
        #             # plt_table[pos].set_facecolor(cmap(memoization[reduced_table['workload'].iloc[pos[1]]]))
        #             cell.set_alpha(0.5)
        #         cell.set_linewidth(0.3)
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
    ax_u.get_legend().remove()
    ax_c.get_legend().remove()
    ax_r.get_legend().remove()

    if simple_visualization:
        ax.minorticks_on()
        ax.yaxis.grid(which='both')
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())

    ax_u.set_ylabel('service\nutilization (%)')
    ax_u.set_ylim(0, 1.0)
    ax_u.set_yticks(np.linspace(0, 1.0, 5))
    rlabels = [f'{item:.0f}' for item in np.linspace(0, 1.0, 5)*100]
    rlabels[-1] += '>'
    ax_u.set_yticklabels(rlabels)

    ax_t.set_ylabel('arrivals/s')
    ax_t.set_ylim(0, ax_t.get_yaxis().get_data_interval()[1])
    ax_t.set_yticks(np.linspace(0, ax_t.get_yaxis().get_data_interval()[1], 4))



    ax_r.set_ylabel('resp.\ntime(s)')
    ax_r.set_ylim(0, .2)
    ax_r.set_yticks(np.linspace(0, .2, 6))
    rlabels = [f'{item:.4f}' for item in np.linspace(0, .2, 6)]
    rlabels[-1] += '>'
    ax_r.set_yticklabels(rlabels)

    ax_m.set_yscale('log', base=2)
    # ax_m.set_ylim(0, ax_m.get_yaxis().get_data_interval()[1])
    ax_m.set_ylim(256, 8192)
    ax_m.set_yticks([256, 512, 1024, 2048, 4096, 8192])
    ax_m.set_yticklabels([256, 512, 1024, 2048, 4096, 8192])
    # ax_m.set_yticks([0, 2**9, 2**10, 2**11, 2**12, 2*13])
    ax_m.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax_m.set_ylabel('memory (MB)\n'+r'${\log_2}$ scale')

    # ax_m.set_ylabel('Mem (%)')
    # ax_m.set_ylim(0, 1)
    # ax_m.set_yticks(np.linspace(0,1,5))
    # ax_m.set_yticklabels([f'{item:.0f}' for item in np.linspace(0, 100, 5)])
    # ax_m.get_yaxis().get_major_formatter().labelOnlyBase = False

    # ax_c.set_ylabel('CPU (%)')
    # ax_c.set_ylim(0, 1)
    # ax_c.set_yticks(np.linspace(0,1,5))
    # ax_c.set_yticklabels([f'{item:.0f}' for item in np.linspace(0, 100, 5)])
    # ax_c.get_yaxis().get_major_formatter().labelOnlyBase = False

    ax_c.set_ylim(0, ax_c.get_yaxis().get_data_interval()[1])
    # ax_c.set_ylim(1, 8192)
    ax_c.set_yticks(list(range(1,int(ax_c.get_yaxis().get_data_interval()[1]),2)))
    ax_c.set_yticklabels(list(range(1,int(ax_c.get_yaxis().get_data_interval()[1]),2)))
    # ax_m.set_yticks([0, 2**9, 2**10, 2**11, 2**12, 2*13])
    ax_c.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax_c.set_ylabel('CPU')

    ax_t.set_title(title, loc='left')
    ax_t.axes.get_xaxis().set_visible(False)
    ax_u.axes.get_xaxis().set_visible(False)
    ax_m.axes.get_xaxis().set_visible(False)
    ax_c.axes.get_xaxis().set_visible(False)
    ax_r.axes.get_xaxis().set_visible(False)
    ax_t.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
    ax_u.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
    ax_m.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
    ax_c.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
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
    ], frameon=False, ncol=5, bbox_to_anchor=(0.6, 1.72), loc='upper center', fontsize='small')

    ax.set_ylabel(objective_label)
    # customize label position, -2 and 30 are magic numbers
    ax.text(-2, 0, 'prod.\ncfg.', fontsize='smaller')
    ax.text(-2, top + magic_number, 'train.\ncfg.', fontsize='smaller')

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

    # print(df.to_csv())

    # ax = df.plot.bar()
    #
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, frameon=False, ncol=5, bbox_to_anchor=(0.5, 1.15), loc='upper center', fontsize='small')
    #
    #
    #
    # plt.show()

def general():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # title = 'Daytrader'
    # service_name = "daytrader-service"
    # config_name = "daytrader-config-app"

    # title= "Quarkus"
    # service_name = "quarkus-service"
    # config_name = "quarkus-cm-app"

    title = "AcmeAir"
    service_name = "acmeair-service"
    config_name = "acmeair-config-app"

    name = 'trace-2021-07-12T23 06 37'
    name = 'trace-2021-07-15T18 27 10'
    name = 'trace-2021-07-16T21 18 09'
    name = 'trace-2021-07-17T18 20 54'
    name = 'trace-2021-07-19T19 44 38'
    name = 'trace-2021-07-21T00 08 55'
    name = 'trace-2021-07-21T15 15 57'
    name = 'trace-2021-07-22T17 57 14'
    name = 'trace-2021-07-24T17 00 11'
    name = 'trace-2021-07-27T23 38 28'
    name = 'trace-2021-07-28T17 37 26'
    name = 'trace-2021-07-29T23 49 32'
    name = 'trace-2021-07-30T17 56 45'
    name = 'trace-2021-07-31T15 05 13' # 900
    name = 'trace-2021-08-01T15 49 02' # 600
    name = 'trace-2021-08-02T15 50 13' # 900 no readiness probe
    name = 'trace-2021-08-03T22 50 32' # 50 clients
    name = 'trace-2021-08-04T23 34 25'
    name = 'trace-2021-08-06T16 14 33'
    name = 'trace-2021-08-09T14 33 34' # svc utilization in workload 2 slowly trends downwards
    # name = 'trace-2021-08-12T15 13 28'
    # name = 'trace-2021-08-13T15 04 31'
    # name = 'trace-2021-08-18T22 41 54'
    name = 'trace-2021-08-19T22 37 44'
    name = 'trace-2021-08-26T02 10 11'
    name = 'trace-quarkus-2021-08-27T04 19 51' # quarkus, multi-workloads (50, 100, 200)
    # name = 'trace-acmeair-2021-08-28T00 21 27' # acmeair, multi-workloads (50, 100, 200)
    name = 'trace-daytrader-2021-08-28T00 23 06' # daytrader (5, 10, 50)
    # name = 'trace-jsp-2021-03-11T13 41 07' # JSP
    # name = 'trace-jsf-2021-03-10T14 01 00' # JSF
    # name = 'trace-acmeair-2021-08-31T00 15 53' #acmeair trinity
    # name = 'trace-acmeair-2021-08-31T19 27 42' # acmeair trinity 08
    # name = 'trace-quarkus-2021-08-31T18 03 04' # quarkus trinity 08
    name = 'trace-acmeair-2021-09-07T20 50 22' # ok
    name = 'trace-acmeair-2021-09-08T17 59 42'

    # JSF
    # plot_importance(data)

    # for workload in [''] + [f'workload_{i}' for i in [5, 10, 50]]:
    for workload in [''] + [f'workload_{i}' for i in [50, 100, 200]]:
        print('workload: ', workload)
        try:
            print(workload)
            # if 'workload_1' != workload:
            #     continue
            df = load_raw_data('./resources/' + name + '.json', service_name, workload,
                               skip_reset=True,
                               skip_pruned=True,
                               skip_tuned=True,
                               show_workload_gap=False,
                               to_gib=False)

            empty = df[(df['pname'].str.len() > 0)]
            if empty.empty:
                print(f'skiping {workload}')
                continue
            plot(df, title=title + ': ' + name + '\n' + workload,
                 # objective_label=r'$(1-error)*\frac{1}{(1+resp. time)} \times \frac{requests}{N \times \$}$',
                 objective_label='score',
                 save=False,
                 simple_visualization=False,
                 show_table=True)
            # break
            # plot_app_curves(df, service_name, config_name)
            # plot_importance(importance(df, 'daytrader', 'config'))
        except:
            traceback.print_exc()
            continue

if __name__ == '__main__':
    general()