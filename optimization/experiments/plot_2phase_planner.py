import hashlib
import json
import re
import os
import sys
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd
import numpy as np
import random
import math

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.table import Table
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from pandas.plotting import scatter_matrix


def reset_seeds():
   np.random.seed(123)
   random.seed(123)

reset_seeds()

SEED=0
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.execv(sys.executable, [sys.executable] + sys.argv)

def load_raw_data(filename:str) -> pd.DataFrame:
    raw_data = []

    def name(data):
        return hashlib.md5(bytes(str(data.items()), 'ascii')).hexdigest()

    with open(filename) as jsonfile:
        for row in jsonfile:
            row = re.sub(r'\{"_id":\{"\$oid":"[a-z0-9]+"\},', '{', row)
            record = json.loads(row)
            raw_data.append(Iteration(
                record['iteration'],
                record['production']['curr_config']['name'],
                # name(record['production']['curr_config']['data']), # generate name on the fly
                math.fabs(record['production']['curr_config']['score']),
                math.fabs(record['production']['curr_config']['stats']['mean']),
                math.fabs(record['production']['curr_config']['stats']['median']),
                math.fabs(record['production']['curr_config']['stats']['min']),
                math.fabs(record['production']['curr_config']['stats']['max']),
                record['production']['curr_config']['stats']['stddev'],
                record['production']['metric']['throughput'],
                record['production']['metric']['process_time'],
                record['production']['metric']['memory'] / 2**20,
                record['production']['metric'].get('memory_limit', record['production']['curr_config']['data']['daytrader-service']['memory'] * 2**20) / 2**20,
                record['production']['metric'].get('restarts', 1),
                Param(record['production']['curr_config']['data'], math.fabs(record['production']['curr_config']['score'])),
                record['training']['curr_config']['name'],
                # name(record['training']['curr_config']['data']), # generate name on the fly
                math.fabs(record['training']['curr_config']['score']),
                math.fabs(record['training']['curr_config']['stats']['mean']),
                math.fabs(record['training']['curr_config']['stats']['median']),
                math.fabs(record['training']['curr_config']['stats']['min']),
                math.fabs(record['training']['curr_config']['stats']['max']),
                record['training']['curr_config']['stats']['stddev'],
                record['training']['metric']['throughput'],
                record['training']['metric']['process_time'],
                record['training']['metric']['memory'] / 2**20,
                record['training']['metric'].get('memory_limit', record['training']['curr_config']['data']['daytrader-service']['memory'] * 2**20) / 2**20,
                record['training']['metric'].get('restarts', 1),
                Param(record['training']['curr_config']['data'], math.fabs(record['training']['curr_config']['score'])),
                {chr(i+97):best['name'][:3] for i, best in enumerate(record['best'])} if 'best' in record else []
            ).__dict__)
    df = pd.DataFrame(raw_data).reset_index()
    return df

class Param:
    def __init__(self, params, score):
        self.params = params
        self.params['score'] = score

class Iteration:
    def __init__(self, iteration,
                 pname, pscore, pmean, pmedian, pmin, pmax, pstddev, ptruput, pproctime, pmem, pmem_lim, prestarts, pparams,
                 tname, tscore, tmean, tmedian, tmin, tmax, tstddev, ttruput, tproctime, tmem, tmem_lim, trestarts, tparams, nbest):
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

def plot(df: pd.DataFrame, title:str,save:bool=False, show_table:bool=False):
    # use generate a color pallete
    from SecretColors.cmaps import ColorMap, TableauMap
    from SecretColors import Palette
    # cm = ColorMap(matplotlib)
    cm = TableauMap(matplotlib)
    # p = Palette('ibm', seed=SEED)
    # my_colors = [p.red(shade=30), p.white(), p.blue(shade=60)]
    # my_colors = p.random_gradient(no_of_colors=10)
    colormap = cm.colorblind()  # cm.from_list(p.random(nrows, seed=SEED))
    # colormap = cm.blue_red()

    reduced_table = df
    memoization = {}
    nrows = len(reduced_table)
    new_colors = []
    for index, row in reduced_table.iterrows():
        punique = row['pname']
        tunique = row['tname']
        p = index

        if not punique in memoization:

            memoization[punique] = abs(hash(punique)) / sys.maxsize
            memoization[punique[:3]] = memoization[punique]

        if punique in memoization:
            new_colors.append(memoization[punique])

        if not tunique in memoization:
            memoization[tunique] = abs(hash(tunique)) / sys.maxsize
            memoization[tunique[:3]] = memoization[tunique]

        if tunique in memoization:
            new_colors.append(memoization[tunique])

    # reduced_table['index'] = [i + 0.5 for i in range(len(reduced_table))]
    # plotting
    fig: Figure
    ax: Axes
    ax_t: Axes
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(nrows=4, hspace=0.01, height_ratios=[1, 1, 1, 4])
    axs = gs.subplots(sharex='col')
    ax = axs[3]
    ax_r = axs[2]
    ax_m = axs[1]
    ax_t = axs[0]
    # fig, axs = plt.subplots(figsize=(18, 8))

    # split chart by configs and paint each region with a unique color
    cmap = matplotlib.cm.get_cmap(colormap)
    k = 3
    count = 1
    top_p = max(reduced_table['pmean']+reduced_table['pstddev'])
    top_t = max(reduced_table['tmean']+reduced_table['tstddev'])
    top = max(max(reduced_table['pmax']), max(reduced_table['tmax']))
    ax.set_ylim(ymax=top+40)
    for index, row in reduced_table.iterrows():
        _tmax = max(row['pmean'], row['tscore'])
        _tmin = min(row['pmean'], row['tscore'])

        _pmax = max(row['pmean'], row['pscore'])
        _pmin = min(row['pmean'], row['pscore'])

        ax.text(index, 0, f"{row['pname'][:3]}\n{row['prestarts']}", {'ha': 'center', 'va': 'bottom'}, rotation=45, fontsize='x-small',
                color='red')
        ax.text(index, top+40, f"{row['tname'][:3]}\n{row['trestarts']}", {'ha': 'center', 'va': 'top'}, rotation=45, fontsize='x-small',
                color='blue')

        ax.axvspan(index - 0.5, index + 0.5, facecolor=cmap(memoization[row['pname']]), alpha=0.5)
        # ax.axvspan(index - 0.5, index + 0.5, facecolor=cmap(memoization[row['pname']]), alpha=0.5-row['prestarts']/len(reduced_table))

        plot_training([index, _tmin], [index, _tmax], [index, row['tscore']], ax, color='blue', marker='x', linestyle='--', linewidth=0.4)
        plot_training([index, _pmin], [index, _pmax], [index, row['pscore']], ax, color='red', marker='None', linestyle='--', linewidth=0.4)
        # add divisions between iterations
        newline_yspan([index+0.5, 0], [index+0.5, top+10], ax)
    newline_yspan([-0.5, 0], [-0.5, top+10], ax)

    kind = 'bar'
    ax_t = reduced_table.plot.bar(ax=ax_t, x='index', y=['ptruput', 'ttruput'], rot=0, color={'ptruput':'red', 'ttruput':'blue'}, width=0.8, alpha=0.7)
    ax_r = reduced_table.plot.bar(ax=ax_r, x='index', y=['pproctime', 'tproctime'], rot=0, color={'pproctime':'red', 'tproctime':'blue'}, width=0.8, alpha=0.7)
    ax_m = reduced_table.plot.bar(ax=ax_m, x='index', y=['pmem', 'tmem'], rot=0, color={'pmem':'red', 'tmem':'blue'}, width=0.8, alpha=1)
    ax_m = reduced_table.plot.bar(ax=ax_m, x='index', y=['pmem_lim', 'tmem_lim'], rot=0, color={'pmem_lim':'red', 'tmem_lim':'blue'}, width=0.8, alpha=0.3)

    ax = reduced_table.plot(ax=ax, x='index', y='pmedian', color='yellow', marker='^', markersize=3, linewidth=0)
    ax = reduced_table.plot(ax=ax, x='index', y='pmean', color='black', marker='o', markersize=3, yerr='pstddev', linewidth=0, elinewidth=0.7, capsize=3)
    ax = reduced_table.plot(ax=ax, x='index', y='tmedian', color='lime', marker='^', markersize=3, linewidth=0)
    ax = reduced_table.plot(ax=ax, x='index', y='pscore', marker='*', markersize=4, color='red', linewidth=0)
    ax.set_ylim(ymin=0)# customize x-ticks

    if not show_table:
        ax.xaxis.set_ticks(range(len(reduced_table)))
        ax.set_xlabel('index')
        ax.margins(x=0)
        ax.tick_params(axis='x', which='major', labelsize='x-small')
        ax.tick_params(axis='x', which='minor', labelsize='x-small')
    else:
        ax.xaxis.set_ticks([])
        ax.set_xlabel('')
        ax.margins(x=0)

        plt_table: matplotlib.table
        table = pd.DataFrame(reduced_table['nbest'].to_dict())
        table = table.fillna(value='')
        plt_table:Table

        plt_table = ax.table(cellText=table.to_numpy().reshape(3,-1), rowLoc='center',
                 rowLabels=['1st','2nd','3rd'], colLabels=reduced_table['index'],
                 # colWidths=[.5,.5],
                 cellLoc='center',
                 colLoc='center', loc='bottom')
        plt_table.set_fontsize('x-small')

        for pos, cell in plt_table.get_celld().items():
            cell.fill = True
            text = cell.get_text().get_text()
            if len(text) == 3 and text not in '1st2nd3rd':
                plt_table[pos].set_facecolor(cmap(memoization[text]))
                cell.set_alpha(0.7)

        old_ax_pos = ax_t.get_position()
        # new_pos = Bbox([[old_ax_pos.x0, old_ax_pos.y0]], [[plt_table.get_po]])
        # ax_t.set_position()
    # reduced_table.plot(table=np.array(reduced_table['nbest'].T), ax=ax)

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

    ax.get_legend().remove()
    ax_m.get_legend().remove()
    ax_r.get_legend().remove()
    # ax.set_title(title, loc='left')
    ax_t.set_ylabel('requests')
    ax_r.set_ylabel('response time')
    # print(ax_m.get_yaxis().get_data_interval())
    ax_t.set_ylim(0, ax_t.get_yaxis().get_data_interval()[1])
    ax_r.set_ylim(0, .5)
    ax_t.set_yticks(np.linspace(0, ax_t.get_yaxis().get_data_interval()[1], 4))
    ax_r.set_yticks(np.linspace(0, .5, 4))
    rlabels = [f'{item:.2f}' for item in np.linspace(0, .5, 4)]
    rlabels[-1] += '>'
    ax_r.set_yticklabels(rlabels)
    ax_m.set_ylim(0, ax_t.get_yaxis().get_data_interval()[1])
    ax_m.set_yticks([0,  1024, 2048, 4096, 8192])
    ax_m.set_ylabel('memory (MB)')
    ax_t.set_title(title, loc='left')
    ax_t.axes.get_xaxis().set_visible(False)
    ax_m.axes.get_xaxis().set_visible(False)
    ax_t.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
    ax_m.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
    ax_r.grid(True, linewidth=0.3, alpha=0.7, color='k', linestyle='-')
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

    ax.set_ylabel('requests/$')

    ax_pos = ax.get_position()
    ax.text(-2.2, 0, 'train.\ncfg.', fontsize='smaller')
    ax.text(-2.2, top + 30, 'prod.\ncfg.', fontsize='smaller')

    gs.tight_layout(fig)
    if save:
        fig = plt.gcf()
        fig.set_size_inches((18, 8), forward=False)
        fig.savefig(save, dpi=150)  # Change is over here
    else:
        plt.show()
    plt.show()


def newline(p1, p2, ax, arrow=False, **kwargs):
    xmin, xmax = ax.get_xbound()

    if arrow:
        if p1[1] > p2[1]:
            ax.scatter(p2[0], p2[1], marker='^', color=kwargs['color'])
        elif p2[1] > p1[1]:
            ax.scatter(p1[0], p1[1], marker='v', color=kwargs['color'])

    l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
    ax.add_line(l)
    return l


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
    ax.scatter(train[0], train[1], marker=marker, color=kwargs['color'])

    top_y = max([train[1], p1[1], p2[1]])
    bot_y = min([train[1], p1[1], p2[1]])

    ax.scatter(train[0], train[1], marker=marker, color=kwargs['color'])


    l = mlines.Line2D([p1[0], p2[0]], [bot_y, top_y], **kwargs)
    ax.add_line(l)
    return l

    #
    # if p1[1] < train[1] < p2[1]:
    #     ax.scatter(train[0], train[1], marker='x', color=kwargs['color'])
    # else:
    #     if max(p1[1], p2[1]) > train[1]:
    #         ax.scatter(train[0], train[1], marker='x', color=kwargs['color'])
    #     else:
    #         ax.scatter(train[0], train[1], marker='x', color=kwargs['color'])
    #
    #     l = mlines.Line2D([p1[0], p2[0]], [train[1], max(p1[1], p2[1])], **kwargs)
    #     ax.add_line(l)
    #     return l

def test():

    arr1 = [{"name":"3628671b941d84573fbf976ec3df444e", "uid":37, "score":-313.05764272920436, "mean":-313.05764272920436, "std":0.0, "median":-313.05764272920436 }, {"name":"d40fcbcf18c040e215c2e581aca5d0e7", "uid":45, "score":-309.1118529457984, "mean":-309.1118529457984, "std":0.0, "median":-309.1118529457984}, {"name":"915f3984fe38cfd2426117867985af04", "uid":23, "score":-288.923972539127, "mean":-288.923972539127, "std":0.0, "median":-288.923972539127}, {"name":"52f951fd9fffd5083688305ff1954c1b", "uid":46, "score":-267.5291577148189, "mean":-267.5291577148189, "std":0.0, "median":-267.5291577148189}, {"name":"a4f40b9c5a8b8180d3270c6c3bdb9e13", "uid":52, "score":-271.797606047275, "mean":-271.797606047275, "std":0.0, "median":-271.797606047275}, {"name":"08865885d8ec21b0b524f68a75ee2371", "uid":21, "score":-273.0127790605311, "mean":-273.0127790605311, "std":0.0, "median":-273.0127790605311}, {"name":"08ee16e81439b290caabfbc434e20760", "uid":24, "score":-249.98036842618419, "mean":-249.98036842618419, "std":0.0, "median":-249.98036842618419}, {"name":"f7de4a87cfc6954d98f76e9e6b34ae9d", "uid":28, "score":-174.8862204359479, "mean":-174.8862204359479, "std":0.0, "median":-174.8862204359479}, {"name":"f2da066bf5376bfcf328c21019a99abd", "uid":6, "score":-284.4293414046439, "mean":-237.35385225537826, "std":122.99436511836835, "median":-260.73477268750173}, {"name":"33b791f490e0d245ed60d77ace4470e5", "uid":32, "score":-237.04651835141965, "mean":-237.04651835141965, "std":0.0, "median":-237.04651835141965}, {"name":"7199c0884bd7039cb97bff4e1c4bbcdf", "uid":55, "score":-222.6024977585152, "mean":-222.6024977585152, "std":0.0, "median":-222.6024977585152}, {"name":"6b913d38b02972980f3964abab05ec77", "uid":35, "score":-136.69385665168468, "mean":-136.69385665168468, "std":0.0, "median":-136.69385665168468}, {"name":"704c8ca96b94a184a3daf09c61b7a35b", "uid":17, "score":-215.28841624547283, "mean":-215.28841624547283, "std":0.0, "median":-215.28841624547283}, {"name":"7a12f74b8d16e7023bde3926fe8f89d3", "uid":16, "score":-115.42601531294116, "mean":-115.42601531294116, "std":0.0, "median":-115.42601531294116}, {"name":"2d02f48746ef64c8389aa4766b3c3f49", "uid":40, "score":-223.65509321647264, "mean":-223.65509321647264, "std":0.0, "median":-223.65509321647264}, {"name":"e59b658ddd6fc4ba6b7298270b4c3876", "uid":43, "score":-98.04738752712257, "mean":-98.04738752712257, "std":0.0, "median":-98.04738752712257}, {"name":"dc3e2b808f3c2a9f12f55f7d56a21fd1", "uid":14, "score":-145.47334208168073, "mean":-145.47334208168073, "std":0.0, "median":-145.47334208168073}, {"name":"b08cea7eb02739564bea343394943ebd", "uid":29, "score":-159.3881544555052, "mean":-159.3881544555052, "std":0.0, "median":-159.3881544555052}, {"name":"8005911ab246a8182432179d5d874919", "uid":27, "score":-152.30010109901625, "mean":-152.30010109901625, "std":0.0, "median":-152.30010109901625}, {"name":"5a49c2ed3864189d014521c142994aa1", "uid":31, "score":-151.2656921766894, "mean":-151.2656921766894, "std":0.0, "median":-151.2656921766894}, {"name":"eca878ad4ea2bf5853b5cab0b3e7673e", "uid":53, "score":-192.62740158780284, "mean":-192.62740158780284, "std":0.0, "median":-192.62740158780284}, {"name":"623e8776e850308015b9560f24ca55e0", "uid":18, "score":-210.50146666254957, "mean":-210.50146666254957, "std":0.0, "median":-210.50146666254957}, {"name":"4579fd533562b3e33a99948f7b9264e6", "uid":57, "score":-81.6937609897022, "mean":-81.6937609897022, "std":0.0, "median":-81.6937609897022}, {"name":"b29bc5f6d5ef1dd06bf02c5164d064ce", "uid":13, "score":-96.43715604835184, "mean":-96.43715604835184, "std":0.0, "median":-96.43715604835184}, {"name":"82cf786b2aa58475485ab59b18aee54d", "uid":34, "score":-115.75547280262052, "mean":-115.75547280262052, "std":0.0, "median":-115.75547280262052}, {"name":"17cdac4f2d13a4e47cf7123efe2a5eef", "uid":36, "score":-74.11731606648115, "mean":-74.11731606648115, "std":0.0, "median":-74.11731606648115}, {"name":"b5dbcdcfb2e962c4bdaa719ca433ca88", "uid":22, "score":-170.3304211834203, "mean":-170.3304211834203, "std":0.0, "median":-170.3304211834203}, {"name":"c9673bbee05491d38e9ed5d99ae859e2", "uid":38, "score":-97.29968703102645, "mean":-97.29968703102645, "std":0.0, "median":-97.29968703102645}, {"name":"c03561371a094732884a5bf9c3538a0c", "uid":39, "score":-99.24349059288835, "mean":-99.24349059288835, "std":0.0, "median":-99.24349059288835}, {"name":"13a7d0828a693e8317f10e269d3d2003", "uid":25, "score":-193.4998255989836, "mean":-193.4998255989836, "std":0.0, "median":-193.4998255989836}, {"name":"f4f88e10eed680e3c43435a1ccbc0cbb", "uid":41, "score":-216.63701827686623, "mean":-216.63701827686623, "std":0.0, "median":-216.63701827686623}, {"name":"6870d90ce01505679580c0c3ad7a9692", "uid":42, "score":-66.12850458080112, "mean":-66.12850458080112, "std":0.0, "median":-66.12850458080112}, {"name":"de734db7e4547fed8bf3a07e1e127c7f", "uid":26, "score":-80.66745396094277, "mean":-80.66745396094277, "std":0.0, "median":-80.66745396094277}, {"name":"976adf780042c641ea23700ebdf68cc6", "uid":12, "score":-83.21484083798681, "mean":-83.21484083798681, "std":0.0, "median":-83.21484083798681}, {"name":"b79c3bced654a14eab87b745cba70810", "uid":44, "score":-85.93120395815805, "mean":-85.93120395815805, "std":0.0, "median":-85.93120395815805}, {"name":"25b204a186e65103748463cc0aaba22c", "uid":19, "score":-134.8803959597834, "mean":-134.8803959597834, "std":0.0, "median":-134.8803959597834}, {"name":"9ad5b300946d035827fa1972e5d36a06", "uid":47, "score":-46.671455534684206, "mean":-46.671455534684206, "std":0.0, "median":-46.671455534684206}, {"name":"bc85cab39eb3eeae2564dc1daf727118", "uid":48, "score":-113.11557898203915, "mean":-113.11557898203915, "std":0.0, "median":-113.11557898203915}, {"name":"8fb6ba10b86be1b4fc39dbb9acb4f5d1", "uid":49, "score":-149.43940605594705, "mean":-149.43940605594705, "std":0.0, "median":-149.43940605594705}, {"name":"846742b5ed17cee88d072eff00954d25", "uid":15, "score":-71.20451414368458, "mean":-71.20451414368458, "std":0.0, "median":-71.20451414368458}, {"name":"185759a8bb5bf46ce5b97468a1fa7005", "uid":51, "score":-62.877347484021826, "mean":-62.877347484021826, "std":0.0, "median":-62.877347484021826}, {"name":"5597b3c76c4984afabdcde1d1e3c8c48", "uid":30, "score":-79.32574315409067, "mean":-79.32574315409067, "std":0.0, "median":-79.32574315409067}, {"name":"f2f8f3f582c8c7938458aed3b1b46bdd", "uid":50, "score":-176.34605446866505, "mean":-176.34605446866505, "std":0.0, "median":-176.34605446866505}, {"name":"ef1ccf0f12c2b6f6921fa89385bcf07e", "uid":20, "score":-140.65691231872026, "mean":-140.65691231872026, "std":0.0, "median":-140.65691231872026}, {"name":"306901bc9efa4cc5a70d51cd4ed067ac", "uid":54, "score":-152.77659020760973, "mean":-152.77659020760973, "std":0.0, "median":-152.77659020760973}, {"name":"06de3441a431f77742d04dc56a531fa0", "uid":33, "score":-58.92511048341837, "mean":-58.92511048341837, "std":0.0, "median":-58.92511048341837}, {"name":"0fb77fc423ef6e02f6c6d495e9cb3175", "uid":56, "score":-72.76731669417943, "mean":-72.76731669417943, "std":0.0, "median":-72.76731669417943}, {"name":"ee019ce98b9ff37a89883a31bef68b16", "uid":58, "score":-61.91288704161767, "mean":-61.91288704161767, "std":0.0, "median":-61.91288704161767}, {"name":"74e18b89c1272b37f031d88e47f7fb72", "uid":59, "score":-76.85348124668894, "mean":-76.85348124668894, "std":0.0, "median":-76.85348124668894}]
    arr2 = [{"name":"f2da066bf5376bfcf328c21019a99abd", "uid":6, "score":-284.4293414046439, "mean":-237.35385225537826, "std":122.99436511836835, "median":-260.73477268750173}]

    import heapq

    heap1 = []
    heap2 = []

    class Config:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __lt__(self, other):
            return self.value < other.value

        def __str__(self):
            return f'{{"name":{self.name}, "value":{self.value}}}'

        def __repr__(self):
            return self.__str__()

    for item in arr1:
        heapq.heappush(heap1, Config(item['name'], item['median']))

    for item in arr2:
        heapq.heappush(heap2, Config(item['name'], item['median']))

    best1 = heapq.nsmallest(1, heap1)
    best2 = heapq.nsmallest(1, heap2)
    best_concat = best1 + best2
    heapq.heapify(best_concat)

    best = heapq.nsmallest(1, best_concat)


def plot_app_curves(df: pd.DataFrame, title:str=''):
    # to_plot = df[[]]
    data = {}
    for index, row in df.iterrows():
        tmp = {}
        # print(row['tparams'].params)
        tmp = row['tparams'].params['daytrader-config-jvm']
        tmp['score'] = row['tparams'].params['score']
        # print(row['tparams'].params)
        # print(tmp)
        # try:
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
        # except:
        #     continue

    df = pd.DataFrame(data)
    # df.plot(y=['HTTP_PERSIST_TIMEOUT'])
    mat_ax = scatter_matrix(df, alpha=0.5, figsize=(6,6), diagonal='kde')
    # xlabel = ax.get_xlabel()
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
    title = 'tDaytrader'

    df = load_raw_data('./resources/'+name+'.json')
    plot(df, title=title+': '+name, save=False, show_table=True)
    # plot_app_curves(df)