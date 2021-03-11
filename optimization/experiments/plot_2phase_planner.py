import hashlib
import json
import math
import os
import random
import re
import sys

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


def load_raw_data(filename: str, service_name: str) -> pd.DataFrame:
    raw_data = []

    def name(data):
        return hashlib.md5(bytes(str(data.items()), 'ascii')).hexdigest()

    with open(filename) as jsonfile:
        for i, row in enumerate(jsonfile):
            # workaround to avoid problems with mongodb id
            row = re.sub(r'{"_id":{"\$oid":"[a-z0-9]+"},', '{', row)

            record = json.loads(row)

            raw_data.append(Iteration(
                record['iteration'],
                record['production']['curr_config']['name'],
                math.fabs(record['production']['curr_config']['score'] or 0),
                math.fabs(record['production']['curr_config']['stats']['mean'] or 0),
                math.fabs(record['production']['curr_config']['stats']['median'] or 0),
                math.fabs(record['production']['curr_config']['stats']['min'] or 0),
                math.fabs(record['production']['curr_config']['stats']['max'] or 0),
                record['production']['curr_config']['stats']['stddev'] or 0,
                record['production']['metric']['throughput'],
                record['production']['metric']['process_time'],
                record['production']['metric']['memory'] / 2 ** 20,
                record['production']['metric'].get('memory_limit',
                                                   record['production']['curr_config']['data'][service_name][
                                                       'memory'] * 2 ** 20) / 2 ** 20,
                record['production']['metric'].get('restarts', 1),
                Param(record['production']['curr_config']['data'] or {},
                      math.fabs(record['production']['curr_config']['score'] or 0)),
                record['training']['curr_config']['name'],
                math.fabs(record['training']['curr_config']['score']),
                math.fabs(record['training']['curr_config']['stats']['mean']),
                math.fabs(record['training']['curr_config']['stats']['median']),
                math.fabs(record['training']['curr_config']['stats']['min']),
                math.fabs(record['training']['curr_config']['stats']['max']),
                record['training']['curr_config']['stats']['stddev'],
                record['training']['metric']['throughput'],
                record['training']['metric']['process_time'],
                record['training']['metric']['memory'] / 2 ** 20,
                record['training']['metric'].get('memory_limit',
                                                 record['training']['curr_config']['data'][service_name][
                                                     'memory'] * 2 ** 20) / 2 ** 20,
                record['training']['metric'].get('restarts', 1),
                Param(record['training']['curr_config']['data'], math.fabs(record['training']['curr_config']['score'])),
                # create an auxiliary table to hold the 3 best config at every iteration
                {chr(i + 97): best['name'][:3] for i, best in enumerate(record['best'])} if 'best' in record else []
            ).__dict__)

    return pd.DataFrame(raw_data).reset_index()


class Param:
    """ Config params holder"""

    def __init__(self, params, score):
        self.params = params
        self.params['score'] = score


class Iteration:
    def __init__(self, iteration,
                 pname, pscore, pmean, pmedian, pmin, pmax, pstddev, ptruput, pproctime, pmem, pmem_lim, prestarts,
                 pparams,
                 tname, tscore, tmean, tmedian, tmin, tmax, tstddev, ttruput, tproctime, tmem, tmem_lim, trestarts,
                 tparams, nbest):
        self.pname = pname
        self.pscore = pscore
        self.pmean = pmean
        self.pmedian = pmedian
        self.pmin = pmax
        self.pmax = pmin
        self.pstddev = pstddev
        self.ptruput = ptruput
        self.pproctime = pproctime
        self.pmem = pmem
        self.pmem_lim = pmem_lim
        self.prestarts = prestarts
        self.pparams = pparams
        self.tname = tname
        self.tscore = tscore
        self.tmean = tmean
        self.tmedian = tmedian
        self.tmin = tmax
        self.tmax = tmin
        self.tstddev = tstddev
        self.ttruput = ttruput
        self.tproctime = tproctime
        self.tmem = tmem
        self.tmem_lim = tmem_lim
        self.tparams = tparams
        self.trestarts = trestarts
        self.iteration = iteration
        self.nbest = nbest


def plot(df: pd.DataFrame, title: str, objective_label: str = '', save: bool = False, show_table: bool = False):
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

    # plotting elements
    fig: Figure
    ax: Axes  # iterations
    ax_r: Axes  # response time row
    ax_m: Axes  # memory row
    ax_t: Axes  # truput row
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(nrows=4, hspace=0.000, height_ratios=[1, 1, 1, 3])
    axs = gs.subplots(sharex='col')
    # bottom-up rows
    ax = axs[3]  # iterations
    ax_r = axs[2]  # ...
    ax_m = axs[1]
    ax_t = axs[0]

    # split chart by configs and paint each region with a unique color
    cmap = matplotlib.cm.get_cmap(colormap)
    # find highest point in iteration row
    top = max(max(reduced_table['pmax']), max(reduced_table['tmax']))
    ax.set_ylim(ymax=top + 40)  # 40 is a magic number for this figsize
    for index, row in reduced_table.iterrows():
        # max min score boundaris at every iteration
        _tmax = max(row['pmean'], row['tscore'])
        _tmin = min(row['pmean'], row['tscore'])

        _pmax = max(row['pmean'], row['pscore'])
        _pmin = min(row['pmean'], row['pscore'])

        # configuration label
        ax.text(index, 0, f"{row['pname'][:3]}", {'ha': 'center', 'va': 'bottom'}, rotation=45, fontsize='x-small',
                color='red')  # production
        ax.text(index, top + 40, f"{row['tname'][:3]}", {'ha': 'center', 'va': 'top'}, rotation=45, fontsize='x-small',
                color='blue')  # training

        # paint full column
        ax.axvspan(index - 0.5, index + 0.5, facecolor=cmap(memoization[row['pname']]), alpha=0.5)

        # draw delta(score, (max,min)) -- residual
        plot_training([index, _tmin], [index, _tmax], [index, row['tscore']], ax, color='blue', marker='x',
                      linestyle='--', linewidth=0.4)
        plot_training([index, _pmin], [index, _pmax], [index, row['pscore']], ax, color='red', marker='None',
                      linestyle='--', linewidth=0.4)

        # add divisions between iterations
        newline_yspan([index + 0.5, 0], [index + 0.5, top + 10], ax)
    newline_yspan([-0.5, 0], [-0.5, top + 10], ax)

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
    ax = reduced_table.plot(ax=ax, x='index', y='pmedian', color='yellow', marker='^', markersize=3, linewidth=0)
    ax = reduced_table.plot(ax=ax, x='index', y='pmean', color='black', marker='o', markersize=3, yerr='pstddev',
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

        plt_table: matplotlib.table
        table = pd.DataFrame(reduced_table['nbest'].to_dict())
        table = table.T
        table['restarts'] = reduced_table['prestarts']
        table = table.T
        table = table.fillna(value='')
        plt_table: Table

        # change restarts line position
        tmp = table.iloc[-1]
        for i in reversed(range(len(table))):
            table.iloc[i] = table.iloc[i - 1]
        table.iloc[0] = tmp

        plt_table = ax.table(cellText=table.to_numpy().reshape(4, -1), rowLoc='center',
                             rowLabels=['r', '1st', '2nd', '3rd'], colLabels=reduced_table['index'],
                             cellLoc='center',
                             colLoc='center', loc='bottom')
        plt_table.set_fontsize('x-small')

        for pos, cell in plt_table.get_celld().items():
            cell.fill = True
            text: str = cell.get_text().get_text()
            if pos[0] != 0 and len(text) == 3 and text not in '1st2nd3rd':
                plt_table[pos].set_facecolor(cmap(memoization[text]))
                cell.set_alpha(0.7)

    # customize legend
    handles, labels = ax.get_legend_handles_labels()
    handles.pop()
    handles.pop()
    handles.pop()
    handles.pop()
    handles.append(
        mlines.Line2D([], [], color='black', marker='o', markersize=4, linestyle='-', linewidth=0.7))
    handles.append(
        mlines.Line2D([], [], color='yellow', marker='^', linestyle='None'))
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
    handles.append(matplotlib.patches.Patch(facecolor='red', edgecolor='k', alpha=0.7,
                                            label='config. color'))
    handles.append(matplotlib.patches.Patch(facecolor='blue', edgecolor='k', alpha=0.7,
                                            label='config. color'))

    # customize y-axis labels
    ax.get_legend().remove()
    ax_m.get_legend().remove()
    ax_r.get_legend().remove()

    ax_t.set_ylabel('requests')
    ax_r.set_ylabel('resp. time (s)')
    ax_t.set_ylim(0, ax_t.get_yaxis().get_data_interval()[1])
    ax_r.set_ylim(0, .5)
    ax_t.set_yticks(np.linspace(0, ax_t.get_yaxis().get_data_interval()[1], 4))
    ax_r.set_yticks(np.linspace(0, .5, 4))
    rlabels = [f'{item:.2f}' for item in np.linspace(0, .5, 4)]
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
        'avg. of config. \'abc\' in prod',
        'median of config \'abc\' in prod',
        'median of config \'abc\' in train',
        'prod. value at n-th iteration',
        'train. value at n-th iteration',
        'residual (*:prod, X:train), $y_i - \overline{Y_i}$',
        'config. \'abc\' color',
        'production',
        'training',
    ], frameon=False, ncol=5, bbox_to_anchor=(0.6, 1.52), loc='upper center', fontsize='small')

    ax.set_ylabel(objective_label)
    # customize label position, -4 and 30 are magic numbers
    ax.text(-4, 0, 'prod.\ncfg.', fontsize='smaller')
    ax.text(-4, top + 30, 'train.\ncfg.', fontsize='smaller')

    # tight layout to avoid waste of white space
    gs.tight_layout(fig)
    if save:
        fig = plt.gcf()
        fig.set_size_inches((18, 8), forward=False)
        fig.savefig(save + '.pdf', dpi=150)  # Change is over here
    else:
        plt.show()
    plt.show()


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


def plot_app_curves(df: pd.DataFrame, title: str = ''):
    data = {}
    for index, row in df.iterrows():
        # extract config only
        tmp = {}
        tmp = row['tparams'].params['daytrader-config-jvm']
        tmp.update(row['tparams'].params['daytrader-config-app'])
        tmp.update(row['tparams'].params['daytrader-service'])
        tmp['score'] = row['tparams'].params['score']
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
    df = pd.DataFrame(data)
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


if __name__ == '__main__':
    # test()
    # exit(0)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # df = load_raw_data('./resources/trace-2020-12-18T20 07 09.720531+00 00.json')
    # df = load_raw_data('./resources/trace-2020-12-20T03 05 43.851173+00 00.json')

    # df = load_raw_data('./resources/trace-2020-12-22T18 18 26.696657+00 00.json')

    # df = load_raw_data('./resources/trace-2020-12-23T05 29 37.213918+00 00.json')
    # df = load_raw_data('./resources/trace-2020-12-24T04 36 49.716721+00 00.json')
    # df = load_raw_data('./resources/trace-2020-12-25T00 03 02.json') # why not t-test
    # df = load_raw_data('./resources/trace-2020-12-26T00 40 35.json')
    # df = load_raw_data('./resources/trace-2020-12-28T00 27 18.json')
    # df = load_raw_data('./resources/trace-2020-12-28T20 30 56.json')
    # df = load_raw_data('./resources/trace-2021-01-02T23 47 40.json')
    # df = load_raw_data('./resources/trace-2021-01-05T19 06 26.json')
    # df = load_raw_data('./resources/trace-2021-01-06T00 41 25.json')
    # df = load_raw_data('./resources/trace-2021-01-06T00 41 25.json')

    name = 'trace-2021-01-07T17 05 39'
    name = 'acme-trace-2021-01-09T00 03 17'
    name = 'acme-trace-2021-01-12T01 39 15'
    name = 'acme-trace-2021-01-12T18 28 15'
    name = 'acme-trace-2021-01-13T02 29 40'
    name = 'trace-2021-01-26T05 28 33'
    name = 'trace-2021-01-27T03 23 48'
    name = 'trace-2021-01-07T17 05 39'
    # name = 'trace-2021-01-28T15 30 14' # 100 extra params
    name = 'trace-2021-01-29T13 01 23'
    name = 'trace-2021-02-02T21 06 02'
    name = 'trace-2021-02-03T14 35 43'
    name = 'trace-2021-02-04T07 57 50'
    name = 'trace-2021-02-05T07 46 33'
    name = 'trace-2021-02-06T00 12 27'
    name = 'trace-2021-02-08T18 38 22'
    name = 'trace-2021-02-09T15 45 19'
    name = 'fruits-trace-2021-02-11T23 13 40'
    name = 'trace-2021-02-11T22 57 03'
    # name = 'fruits-trace-2021-02-12T17 42 03'
    # name = 'fruits-trace-2021-02-12T19 20 26'
    name = 'dtrace-2021-02-17T18 49 43'
    name = 'dtrace-2021-02-24T15 18 34'
    name = 'tdtrace-2021-02-24T22 22 36'
    name = 'tdtrace-2021-02-26T15 32 37'
    name = 'tdtrace-2021-02-28T17 17 26'
    name = 'browsing-trading-trace-2021-03-04T17 12 43'
    name = 'browsing-trading-trx-trace-2021-03-04T17 12 43' # browsing-trading
    # name = 'trading-browsing-trx-trace-2021-03-04T17 12 43'
    name = 'trace-trading-2021-03-08T16 54 14'
    name = 'trace-jsf-2021-03-10T14 01 00'
    title = 'Daytrader'
    # title = 'Fruits'
    service_name = "daytrader-service"
    # service_name = 'quarkus-service'
    df = load_raw_data('./resources/' + name + '.json', service_name)
    # plot(df, title=title+': '+name, objective=r'$\frac{1}{1+resp. time} \times \frac{requests}{\$}$', save=title+name, show_table=True)
    plot(df, title=title + ': ' + name, objective_label=r'$\frac{1}{(1+resp. time)} \times \frac{requests}{\$}$',
         save=False,
         show_table=True)
    # plot_app_curves(df)
