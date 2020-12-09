import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import hashlib
import os, sys
import math
import heapq
from pprint import pprint
from typing import *

def reset_seeds():
   np.random.seed(123)
   random.seed(123)

reset_seeds()

SEED=0
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.execv(sys.executable, [sys.executable] + sys.argv)

def load_configs(filename:str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df.fillna(0)

    tconfigs_names = [name for name in df.columns if name.startswith('config_to_eval')]
    treduced_table = df[tconfigs_names]

    instances = []
    for row in treduced_table.iterrows():
        uid = hashlib.md5(bytes(str(tuple(row[1:][0][1:])), 'ascii')).hexdigest()
        instances.append(
            Instance(
                name=uid,
                idx=row[0],
                seed=df['config_to_eval.daytrader-service.memory'].iloc[row[0]],
                value=math.fabs(df['training_metric.objective'].iloc[row[0]]),
                speed_inclination=125
            ))

    configs_names = [name for name in df.columns if name.startswith('last_config')]
    configs_table = df[configs_names]

    pinstances = []
    for row in configs_table.iterrows():
        uid = hashlib.md5(bytes(str(tuple(row[1:][0][1:])), 'ascii')).hexdigest()
        pinstances.append(
            Instance(
                name=uid,
                idx=row[0],
                seed=df['config_to_eval.daytrader-service.memory'].iloc[row[0]],
                value=math.fabs(df['training_metric.objective'].iloc[row[0]]),
                speed_inclination=125
            ))

    return instances, pinstances[1]

class Instance:
    def __init__(self, name:str, idx:int, seed:int, value:float, speed_inclination:float = 100):
        self.speed_inclination = speed_inclination
        self.name = name
        self.idx = idx
        self.seed = seed
        self.value = value
        self.current_value = self.value
        self.it = 0
        self.tuning_iterations = 1

    def __lt__(self, other):
        return self.value < other.value

    def clone(self):
        return Instance(name=self.name, idx=self.idx, seed=self.seed, value=self.value, speed_inclination=self.speed_inclination)

    def __str__(self):
        return f'(idx:{self.idx}, name:{self.name}, seed:{self.seed}, value:{self.value}'

    def __repr__(self):
        return f'(idx:{self.idx}, name:{self.name}, seed:{self.seed}, value:{self.value}'

    def __eq__(self, other):
        return self.name == other.name

    def next_value(self):
        m = 1.0 - self.seed/10240
        self.current_value = -m * self.it * self.speed_inclination + self.value * random.uniform(0.9, 1.1)
        self.current_value = self.current_value if self.current_value >= 0 else 0
        self.it += 1
        return self.current_value

    def reset(self):
        self.it = 0

def plot(history:List[Tuple[float, Instance]], plot_std=False, exp=True, r=0.8, save:str=''):
    # use generate a color pallete
    from SecretColors.cmaps import ColorMap, TableauMap
    from SecretColors import Palette
    # cm = ColorMap(matplotlib)
    cm = TableauMap(matplotlib)
    # p = Palette('ibm', seed=SEED)
    # my_colors = [p.red(shade=30), p.white(), p.blue(shade=60)]
    # my_colors = p.random_gradient(no_of_colors=10)
    colormap = cm.colorblind()  # cm.from_list(p.random(nrows, seed=SEED))

    reduced_table = pd.DataFrame.from_dict(history)

    memoization = {}
    nrows = len(reduced_table)
    new_colors = []
    for index, row in reduced_table.iterrows():
        unique = row['pname']

        if not unique in memoization:
            memoization[unique] = abs(hash(unique)+row['it']) / sys.maxsize

        if unique in memoization:
            new_colors.append(memoization[unique])

    for index, row in reduced_table.iterrows():
        unique = row['tname']

        if not unique in memoization:
            memoization[unique] = abs(hash(unique)+row['it']) / sys.maxsize

        if unique in memoization:
            new_colors.append(memoization[unique])

    # max_table = reduced_table.groupby('pname')['pvalue'].max()
    # min_table = reduced_table.groupby('pname')['pvalue'].min()
    # mean_table = reduced_table.groupby('pname')['pvalue'].mean()
    #
    # reduced_table = pd.merge(reduced_table, max_table, how='left', on='pname').rename(columns={'pvalue_x':'pvalue', 'pvalue_y':'max'})
    # reduced_table = pd.merge(reduced_table, min_table, how='left', on='pname').rename(columns={'pvalue_x':'pvalue', 'pvalue_y':'min'})
    # reduced_table = pd.merge(reduced_table, mean_table, how='left', on='pname').rename(columns={'pvalue_x':'pvalue', 'pvalue_y':'mean'})

    reduced_table['max'] = [float('nan') for _ in range(nrows)]
    reduced_table['min'] = [float('nan')  for _ in range(nrows)]
    reduced_table['mean'] = [float('nan')  for _ in range(nrows)]
    reduced_table['std'] = [float('nan')  for _ in range(nrows)]

    last_row = None
    # rs = RunningStats(a=r)
    for index, it_row in reduced_table.iterrows():
        row = it_row.copy()

        if last_row is None or row['pname'] != last_row['pname']:
            rs = RunningStats(a=r)

        rs.push(row['pvalue'])
        row['std'] = rs.standard_deviation()
        mean = rs.exp_mean() if exp else rs.mean()
        row['mean'] = mean
        if plot_std:
            row['max'] = mean + rs.standard_deviation()
            row['min'] = mean - rs.standard_deviation()
        else:
            row['max'] = rs.max()
            row['min'] = rs.min()


        last_row = row
        reduced_table.at[index] = row

    # adjust postition at x-axis
    reduced_table['iterations'] = [i + 0.5 for i in range(len(reduced_table))]
    # plotting
    ax = reduced_table.plot.scatter(x='iterations', y='max', color='black', marker='_')
    ax = reduced_table.plot.scatter(ax=ax, x='iterations', y='min', color='black', marker='_')
    ax = reduced_table.plot.scatter(ax=ax, x='iterations', y='mean', color='black', marker='o')
    ax = reduced_table.plot.scatter(ax=ax, x='iterations', y='pvalue', marker='*', color='red')

    # split chart by configs and paint each region with a unique color
    cmap = matplotlib.cm.get_cmap(colormap)
    k = 3
    count = 1
    top = max(reduced_table['max']) + 18

    for it, x, yp, yt, c, tc, _min, _max, mean, std in zip(
            reduced_table['it'],
            reduced_table['iterations'],
            reduced_table['pvalue'],
            reduced_table['tvalue'],
            reduced_table['pname'],
            reduced_table['tname'],
            reduced_table['min'],
            reduced_table['max'],
            reduced_table['mean'],
            reduced_table['std']):
        ax.text(x, -20 + 2.5 * ((-1) ** count), c[:3], {'ha': 'center', 'va': 'bottom'}, rotation=0, fontsize='smaller',
                color='red')
        ax.text(x, top - 2.5 * ((-1) ** count), tc[:3], {'ha': 'center', 'va': 'top'}, rotation=0, fontsize='smaller',
                color='blue')
        ax.axvspan(x - 0.5, x + 0.5, facecolor=cmap(memoization[c]), alpha=0.5)
        plot_training([x, _min], [x, _max], [x, yt], ax, color='blue', linestyle='--', linewidth=0.7)
        # newline([x, mean-std], [x, mean+std], ax, color='black', linestyle='-', linewidth=0.7)
        newline([x, _min], [x, _max], ax, color='black', linestyle='-', linewidth=0.7)

        count += 1

    # add divisions at every tuning applied
    for it, istuned in enumerate(reduced_table['tuned']):
        if istuned:
            # if it + 1 < len(reduced_table) and reduced_table.iloc[it]['tuned'] != reduced_table.iloc[it + 1]['tuned']:
            newline_yspan([it + 1, 0], [it + 1, 10], ax)

    # customize x-ticks
    ax.xaxis.set_ticks([])
    # Hide major tick labels
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    # Customize minor tick labels
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([i + 0.5 for i in range(nrows)]))
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
        'std. dev.' if plot_std else 'max-min',
        'config. \'abc\' color',
        # 'train. > prod.'
    ], frameon=True, bbox_to_anchor=(1, 1.08), loc='center', fontsize='small')
    ax.set_title(f'Daytrader simulation: {save}', loc='left')
    ax.set_ylabel('requests/$')

    plt.text(0.1, 0.86, 'train.\nconfig.', fontsize='smaller', transform=plt.gcf().transFigure)
    plt.text(0.1, 0.11, 'prod.\nconfig.', fontsize='smaller', transform=plt.gcf().transFigure)

    # if save:
    #     fig = plt.gcf()
    #     fig.set_size_inches((18, 8), forward=False)
    #     fig.savefig(save, dpi=150)  # Change is over here
    # else:
    #     plt.show()
    plt.show()

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

def update_instance_value(priority_queue:list, instance:Instance, new_value:float):
    items_to_delete = []
    items_to_add = []

    for item in priority_queue:
        if item[1] == instance:
            new_item = (-new_value, item[1])
            items_to_delete.append(item)
            items_to_add.append(new_item)
            break

    for item in items_to_delete:
        priority_queue.remove(item)

    if len(items_to_add) == 0:
        items_to_add.append((-new_value, instance))

    for item in items_to_add:
        heapq.heappush(priority_queue, item)



def classical_tuning(production:Instance, training_list:List[Instance]) -> dict:
    priority_queue = []

    history = []

    prod = production
    for i, training in enumerate(training_list):
        # update value of production when it repeating
        update_instance_value(priority_queue, prod, prod.current_value)

        heapq.heappush(priority_queue, (-training.current_value, training))

        best = priority_queue[0]

        if best[1].name != prod.name and best[1].current_value > prod.current_value:
            prod = best[1]
            prod.reset()
        history.append({'pname':prod.name, 'pvalue':prod.current_value, 'tname':training.name,
                        'tvalue':training.current_value, 'it':prod.tuning_iterations,
                        'tuned':False})

        history[i-1]['tuned'] = history[i - 1]['pname'] != prod.name


        prod.next_value()
        prod.tuning_iterations += 1
    return history

def new_tuning(production:Instance, training_list:List[Instance], k=3) -> dict:
    priority_queue = []
    history = []

    prod = production
    for i, training in enumerate(training_list):

        # --------
        # update value of production when it repeating
        # --------
        update_instance_value(priority_queue, prod, prod.current_value)
        update_instance_value(priority_queue, training, training.current_value)
        # --------
        history.append({'pname': prod.name, 'pvalue': prod.current_value, 'tname': training.name,
                        'tvalue': training.current_value, 'it': prod.tuning_iterations,
                        'tuned': False})
        history[i - 1]['tuned'] = history[i - 1]['pname'] != prod.name

        best = priority_queue[0]
        # at every k-th iteration, k > 0
        just_reset = False
        if i >= k and i % k == 0:
            # update if worth it
            if best[1] != prod:
                prod = best[1]
                prod.reset()
                just_reset = True

        if not just_reset:
            # doesn't move cursor if just reset
            prod.next_value()

    return history

def new_tuning_mean(production:Instance, training_list:List[Instance], k=3, mean_ratio=2) -> dict:
    priority_queue = []
    history = []

    prod = production
    acc = 0
    for i, training in enumerate(training_list):
        if prod.current_value <= mean_ratio * acc:
            # restart if decaying
            prod.reset()
        acc += (prod.current_value - acc) / prod.tuning_iterations

        # --------
        # update value of production when it repeating
        # --------
        update_instance_value(priority_queue, prod, acc)
        update_instance_value(priority_queue, training, training.current_value)
        # --------
        history.append({'pname': prod.name, 'pvalue': prod.current_value, 'tname': training.name,
                        'tvalue': training.current_value, 'it': prod.tuning_iterations,
                        'tuned': False})
        history[i - 1]['tuned'] = history[i - 1]['pname'] != prod.name

        best = priority_queue[0]
        # at every k-th iteration, k > 0
        just_reset = False
        if i >= k and i % k == 0:
            # update if worth it
            if best[1] != prod:
                acc = 0 # restart mean accumulator
                prod.tuning_iterations = 1  # restart mean counter
                prod = best[1]
                prod.reset()
                just_reset = True

        if not just_reset:
            # doesn't move cursor if just reset
            prod.next_value()
            prod.tuning_iterations += 1

    return history

def new_tuning_mean_2phases(production:Instance, training_list:List[Instance], k=3, mean_ratio=2) -> dict:
    priority_queue = []
    history = []

    prod = production
    acc = 0
    counter = 0
    for i, training in enumerate(training_list):
        if counter >= len(training_list): break
        counter += 1
        acc += (prod.current_value - acc) / prod.tuning_iterations
        if prod.current_value <= mean_ratio * acc:
            # restart if decaying
            prod.reset()

        # --------
        # update value of production when it repeating
        # --------
        update_instance_value(priority_queue, prod, acc)
        update_instance_value(priority_queue, training, training.current_value)
        # --------
        history.append({'pname': prod.name, 'pvalue': prod.current_value, 'tname': training.name,
                        'tvalue': training.current_value, 'it': prod.tuning_iterations,
                        'tuned': False})
        history[i - 1]['tuned'] = history[i - 1]['pname'] != prod.name

        best = priority_queue[0]
        # at every k-th iteration, k > 0

        if i >= k and i % k == 0:
            # update if worth it
            if best[1].name != prod.name:
                # saving old values
                old_prod = prod.clone()
                old_prod.tuning_iterations = prod.tuning_iterations
                old_acc = acc

                # reseting
                acc = 0  # restart mean accumulator
                prod.tuning_iterations = 1  # restart mean counter
                prod = best[1]
                prod.reset()

                training = prod.clone()
                for _ in range(k):
                    counter += 1
                    acc += ((prod.current_value+training.current_value)/2 - acc) / prod.tuning_iterations
                    if prod.current_value <= mean_ratio * acc:
                        # restart if decaying
                        prod.reset()

                    if training.current_value <= mean_ratio * acc:
                        # restart if decaying
                        training.reset()

                    # --------
                    # update value of production when it repeating
                    # --------
                    update_instance_value(priority_queue, prod, acc)
                    update_instance_value(priority_queue, training, acc)
                    # --------
                    history.append({'pname': prod.name, 'pvalue': prod.current_value, 'tname': training.name,
                                    'tvalue': training.current_value, 'it': prod.tuning_iterations,
                                    'tuned': False})
                    history[i - 1]['tuned'] = history[i - 1]['pname'] != prod.name

                    prod.next_value()
                    prod.tuning_iterations += 1
                    training.next_value()
                    training.tuning_iterations += 1

                new_best = priority_queue[0]
                # if new_best[1] != prod:
                #     prod = old_prod
                #     acc = old_acc
                # else:
                #     continue

        # doesn't move cursor if just reset
        prod.next_value()
        prod.tuning_iterations += 1

    return history

def new_tuning_expmean_2phases(production:Instance, training_list:List[Instance], k=3, exp=True, r=0.8) -> dict:
    priority_queue = []
    priority_queue_2p = []
    history = []

    prod = production
    counter = 0
    rs_outter = RunningStats(a=r)
    for i, training in enumerate(training_list):
        if counter >= len(training_list): break
        counter += 1
        rs_outter.push(prod.current_value)
        # print(rs_outter.mean(), rs_outter.exp_mean(), rs_outter.standard_deviation())

        mean = rs_outter.exp_mean() if exp else rs_outter.mean()
        # restart pod if current value is out of interest region (std dev limits)
        if mean - prod.current_value > rs_outter.standard_deviation():
            # restart if decaying
            prod.reset()

        # --------
        # update value of production when it repeating
        # --------
        update_instance_value(priority_queue, prod, mean)
        update_instance_value(priority_queue, training, training.current_value)
        # --------
        history.append({'pname': prod.name, 'pvalue': prod.current_value, 'tname': training.name,
                        'tvalue': training.current_value, 'it': prod.tuning_iterations,
                        'tuned': False})
        history[i - 1]['tuned'] = history[i - 1]['pname'] != prod.name

        # [0][0] == current_value

        best = priority_queue_2p[0] if len(priority_queue_2p) > 0 and priority_queue_2p[0][0] < priority_queue[0][0] - rs_outter.standard_deviation() else priority_queue[0]
        # at every k-th iteration, k > 0
        if i >= k and i % k == 0:
            # update if worth it
            if best[1].name != prod.name and best[1].current_value >= (mean + rs_outter.standard_deviation()):
                # discard last batch of iterations
                priority_queue = []
                # keep only the best config from last batch
                heapq.heappush(priority_queue_2p, best)
                # saving old values
                old_prod = prod.clone()
                old_prod.tuning_iterations = prod.tuning_iterations

                # reseting
                prod = best[1]
                prod.tuning_iterations = 1  # restart mean counter
                prod.reset()

                training = prod.clone()
                rs_inner = RunningStats(a=r)
                for j in range(k):
                    counter += 1
                    rs_inner.push((prod.current_value + training.current_value)/2)

                    mean_inner = rs_inner.exp_mean() if exp else rs_inner.mean()
                    # restart pod if current value is out of interest region (std dev limits)
                    if mean_inner - prod.current_value > rs_inner.standard_deviation():
                        # restart if decaying
                        prod.reset()

                    # restart pod if current value is out of interest region (std dev limits)
                    if mean_inner - training.current_value > rs_inner.standard_deviation():
                        # restart if decaying
                        training.reset()

                    # --------
                    # update value of production when it repeating
                    # --------
                    update_instance_value(priority_queue, prod, mean_inner)
                    update_instance_value(priority_queue, training, mean_inner)
                    # --------
                    history.append({'pname': prod.name, 'pvalue': prod.current_value, 'tname': training.name,
                                    'tvalue': training.current_value, 'it': prod.tuning_iterations,
                                    'tuned': False})
                    history[i - 1]['tuned'] = history[i - 1]['pname'] != prod.name

                    prod.next_value()
                    prod.tuning_iterations += 1
                    training.next_value()
                    training.tuning_iterations += 1

                # different configs
                if old_prod.name != prod.name:
                    # statistical comparison using t-test
                    equals, ttest = rs_inner.t_test(rs_outter)
                    # equals === accepts null hypothesis they are equals
                    # ttest >= 0  === caller.mean() >= callee.mean()
                    if equals or ttest >= 0:
                        # only reverts if distributions are different and caller.mean() < callee.mean()
                    # if ttest >= 0:
                    #     # only reverts if caller.mean() < callee.mean(); doens't considers that equals distributions is likely to have same mean in long run
                        rs_outter = rs_outter + rs_inner
                    else:
                        # reverts if new config is worse
                        prod = old_prod
                else:
                    rs_outter = rs_outter + rs_inner



        # doesn't move cursor if just reset
        prod.next_value()
        prod.tuning_iterations += 1
    print()
    return history

from scipy.stats import t
class RunningStats:
    """ https://www.johndcook.com/blog/standard_deviation/ """
    def __init__(self, a=0.8):
        self._a = a
        self._m_n = 0
        self._m_oldM = 0 # normal mean
        self._m_newM = 0
        self._m_oldS = 0 # standard deviation
        self._m_newS = 0
        self._m_newE = 0 # exponential mean
        self._m_oldE = 0

        self._max = float('-inf')
        self._min = float('inf')

    def __add__(self, other):
        rs = RunningStats(a=(self._a + other._a)/2)

        for _ in range(other.n()):
            rs.push(other.mean())

        return rs

    def push(self, x):
        self._m_n += 1

        self._max = max(self._max, x)
        self._min = min(self._min, x)

        if self._m_n == 1:
            self._m_oldE = self._m_newE = self._m_oldM = self._m_newM = x
            self._m_oldS = 0
        else:
            self._m_newE = self._m_oldE * (1-self._a) + x * self._a
            self._m_newM = self._m_oldM + (x - self._m_oldM) / self._m_n
            self._m_newS = self._m_oldS + (x - self._m_oldM) * (x - self._m_newM)

            self._m_oldE = self._m_newE
            self._m_oldM = self._m_newM
            self._m_oldS = self._m_newS

    def n(self):
        return  self._m_n

    def mean(self):
        return self._m_newM if self._m_n > 0 else 0

    def exp_mean(self):
        return self._m_newE if self._m_n > 0 else 0

    def variance(self):
        return self._m_newS / (self._m_n - 1) if self._m_n > 1 else 0

    def standard_deviation(self):
        return math.sqrt(self.variance())

    def max(self):
        return self._max

    def min(self):
        return self._min

    def t_test(self, other, alpha=0.05):
        """https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/"""
        # means
        mean1, mean2 = self.mean(), other.mean()
        # std deviations
        se1, se2 = self.standard_deviation(), other.standard_deviation()
        # standard error on the difference between the samples
        sed = math.sqrt(se1**.0 + se2**2.0)

        # calculate the t statistic
        t_stat = (mean1 - mean2) / sed
        # degrees of freedom
        df = self.n() + other.n() - 2
        # calculate the critical value
        cv = t.ppf(1.0 - alpha, df)
        # calculate the p-value

        p = (1.0 - t.cdf(abs(t_stat), df)) * 2

        # print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
        # # interpret via critical value
        # if abs(t_stat) <= cv:
        #     print('Accept null hypothesis that the means are equal.')
        # else:
        #     print('Reject the null hypothesis that the means are equal.')
        # # interpret via p-value
        # if p > alpha:
        #     print('Accept null hypothesis that the means are equal.')
        # else:
        #     print('Reject the null hypothesis that the means are equal.')

        # return t_stat, df, cv, p
        return p > alpha, t_stat


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    instances, prod = load_configs('resources/logging-trxrhel-202012021945.csv')

    import copy

    # history = classical_tuning(prod.clone(), copy.deepcopy(instances))
    # plot(history, save='classical-st-algorithm.png')
    # history = new_tuning(prod.clone(), copy.deepcopy(instances), k=3)
    # plot(history, save='new_tuning-k=03.png')
    # history = new_tuning(prod.clone(), copy.deepcopy(instances), k=10)
    # plot(history, save='new_tuning-k=10.png')
    # history = new_tuning_mean(prod.clone(), copy.deepcopy(instances), k=3, mean_ratio=1)
    # plot(history, save='new_tuning_mean-k=03-mean=1.png')
    # history = new_tuning_mean(prod.clone(), copy.deepcopy(instances), k=3, mean_ratio=0.9)
    # plot(history, save='new_tuning_mean-k=03-mean=09.png')
    # history = new_tuning_mean(prod.clone(), copy.deepcopy(instances), k=10, mean_ratio=1)
    # plot(history, save='new_tuning_mean-k=10-mean=1.png')
    # history = new_tuning_mean(prod.clone(), copy.deepcopy(instances), k=10, mean_ratio=0.9)
    # plot(history, save='new_tuning_mean-k=10-mean=09.png')
    # history = new_tuning_mean_2phases(prod.clone(), copy.deepcopy(instances), k=3, mean_ratio=1)
    # plot(history, save='new_tuning_mean_2phases-k=03-mean=1.png')
    # history = new_tuning_mean_2phases(prod.clone(), copy.deepcopy(instances), k=3, mean_ratio=0.9)
    # plot(history, save='new_tuning_mean_2phases-k=03-mean=09.png')
    # history = new_tuning_mean_2phases(prod.clone(), copy.deepcopy(instances), k=10, mean_ratio=1)
    # plot(history, save='new_tuning_mean_2phases-k=10-mean=1.png')
    # history = new_tuning_mean_2phases(prod.clone(), copy.deepcopy(instances), k=10, mean_ratio=0.9)
    # plot(history, save='new_tuning_mean_2phases-k=10-mean=09.png')
    exp = False
    r = 0.5
    for i in range(3):
        if i > 0:
            random.shuffle(instances)
        history = new_tuning_expmean_2phases(prod.clone(), copy.deepcopy(instances), k=3, exp=exp, r=r)
        plot(history, plot_std=True, exp=exp, r=r, save='new_tuning_expmean_2phases-k=03.png')

        history = new_tuning_expmean_2phases(prod.clone(), copy.deepcopy(instances), k=10, exp=exp, r=r)
        plot(history,  plot_std=True, exp=exp, r=r, save='new_tuning_expmean_2phases-k=10.png')