import copy
from datetime import datetime, timedelta
from pprint import pprint
import re
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
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

            if 'Tuned' in record['status']:
                continue

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
            table['cost'].append((table['replicas'][-1] * 0.005283024 + (
                    table['memory'][-1] / 1024) * 0.018490583) * 60 / iteration_lenght_minutes)
            # table['cost'].append((table['replicas'][-1] * 0.005283024 + (table['memory'][-1] / 1024) * 0.018490583) * 60 / (table['replicas'][-1]* iteration_lenght_minutes))
            try:
                table['cost_req'].append((table['cost'][-1] * table['replicas'][-1]) / (table['throughput'][-1]))
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
            table['t_cost'].append((table['t_replicas'][-1] * 0.005283024 + (
                    table['t_memory'][-1] / 1024) * 0.018490583) * 60 / iteration_lenght_minutes)
            try:
                table['t_cost_req'].append(table['t_cost'][-1] / (table['t_throughput'][-1]))
            except ZeroDivisionError:
                table['t_cost_req'].append(table['t_cost'][-1])

    df = pd.DataFrame(table)
    # print(df)
    return df


def cost2(df_acme: pd.DataFrame, df_daytrader: pd.DataFrame, df_qhd: pd.DataFrame, df_frameworks: pd.DataFrame,
          cost_per_replica=True):
    for _i, _df in enumerate([df_acme, df_qhd, df_daytrader, df_frameworks]):
        grouped: pd.core.groupby.generic.DataFrameGroupBy = _df[
            ['workload', 'cost', 't_cost', 'cost_req', 't_cost_req']].groupby('workload')
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
            return a * x + b

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
                df['production'] = df['cost'].cumsum()
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
                    value = objective_prod(l + row, ap, bp)
                    extra_production.append(value)
                    no_tuning_value = objective(l + row, a, b)
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
                df['production'] = df['cost_req']
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


def real_cost(tuned: pd.DataFrame, non_tuned: pd.DataFrame, title: '', duration_iteration: int=5, simulated_non_tuning=False):
    grouped: pd.core.groupby.generic.DataFrameGroupBy = tuned[
        ['workload', 'cost', 't_cost', 'cost_req', 't_cost_req']].groupby('workload')

    grouped_non_tuned: pd.core.groupby.generic.DataFrameGroupBy = non_tuned[
        ['workload', 'cost', 't_cost', 'cost_req', 't_cost_req']].groupby('workload')

    fig, axes = plt.subplots(nrows=len(grouped),
                             # figsize=(6,6),
                             ncols=1,
                             sharex='all')

    for _groups in grouped:


        i, groups = _groups
        df = groups.reset_index(drop=True)
        df['production'] = df['cost'].cumsum()
        df['training'] = df['t_cost'].cumsum()

        end_of_tuning_index = (len(df), df['training'].iloc[-1])

        if i not in grouped_non_tuned.groups.keys():
            continue

        workloads = {
            'workload_50': axes[0],
            'workload_100': axes[1],
            'workload_200': axes[2] if len(axes) > 2 else None,
            'workload_jsp': axes[0],
            'workload_jsf': axes[1]
        }
        if title.lower() == 'daytrader':
            workloads = {
                'workload_5': axes[0],
                'workload_10': axes[1],
                'workload_50': axes[2] if len(axes) > 2 else None,
                'workload_jsp': axes[0],
                'workload_jsf': axes[1]
            }
        ax = workloads[i]

        if simulated_non_tuning:
            df['no tuning'] = grouped_non_tuned.get_group(i)['cost'].iloc[:10].cumsum().reset_index(drop=True)
        else:
            df['no tuning'] = grouped_non_tuned.get_group(i)['cost'].cumsum().reset_index(drop=True)
        df = df[['production', 'training', 'no tuning']]

        def objective(x, a, b):
            return a * x + b

        def production_objective(x, a, b):
            return a * x + b
            # return a * np.log(x) + b

        non_na_df = df.dropna()
        df = df.reset_index(drop=True)
        popt, _ = curve_fit(objective, non_na_df.index, non_na_df['no tuning'])
        a, b = popt

        df['no tuning'] = df.apply(
            lambda row: objective(row.name, a, b) if math.isnan(row['no tuning']) else row['no tuning'], axis=1)

        def payoff(production, no_tuning, training):
            return (no_tuning - production) - training

        df['payoff'] = payoff(df['production'], df['no tuning'], df['training'])

        df = df.reset_index(drop=True)
        popt, _ = curve_fit(production_objective, df.index[-10:], df['production'].iloc[-10:])
        ap, bp = popt
        count = 0
        index = len(df)

        profit_index = (0,0)
        # while True:
        while count < 100:
            new_data = {'production': production_objective(index + count, ap, bp),
                        'training': 0,
                        'no tuning': objective(index + count, a, b),
                        }
            count += 1
            new_data['payoff'] = payoff(new_data['production'], new_data['no tuning'], new_data['training'])
            new_data['training'] = float('nan')
            df = df.append(new_data, ignore_index=True)
            if new_data['payoff'] >= new_data['production']:
                profit_index = (len(df), new_data['payoff'])
                break

        df = df.reset_index(drop=True)
        start_payoff_index = \
            [(i, row['payoff']) for i, row in df[['payoff', 'production']].iterrows() if row['payoff'] >= 0][0]

        ax: Axes = df[['production', 'training', 'no tuning', 'payoff']].plot(ax=ax, linewidth=1)

        print(end_of_tuning_index, start_payoff_index, profit_index)

        # make extrapolated points dashed
        line: Line2D
        for line in ax.get_lines():
            dashed_line: Line2D = copy.copy(line)
            dashed_line.set_xdata(dashed_line.get_xdata()[end_of_tuning_index[0]-1:])
            line.set_xdata(line.get_xdata()[:end_of_tuning_index[0]])
            dashed_line.set_ydata(dashed_line.get_ydata()[end_of_tuning_index[0]-1:])
            line.set_ydata(line.get_ydata()[:end_of_tuning_index[0]])
            dashed_line.set_dashes([6, 2])
            dashed_line.set_label('')
            ax.add_line(dashed_line)

        ax.set_xlim(0, df.index[-1]+1)
        ax.set_ylim(df.min().min(), df.max().max())

        yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 4)
        yticks = np.append(yticks, 0)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{round(tick)}' for tick in yticks])

        # ax.vlines(x=end_of_tuning_index[0]-.5, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='k', linewidth=0.7)
        ax.plot(start_payoff_index[0], start_payoff_index[1], 'k*', label='payoff >= 0')
        ax.plot(profit_index[0]-1, profit_index[1], 'kx', label='payoff >= production cost')
        ax.plot(end_of_tuning_index[0]-1, end_of_tuning_index[1], 'k.', label='end of tuning')

        # ax.annotate('end of tuning',  # this is the text
        #             (end_of_tuning_index[0], ax.get_ylim()[1]),  # these are the coordinates to position the label
        #             # textcoords="offset points",  # how to position the text
        #             # xytext=(end_of_tuning_index[0], end_of_tuning_index[1]),  # distance from text to points (x,y)
        #             fontsize='small',
        #             ha='right')  # horizontal alignment can be left, right or center

        xticks = 20
        ax.set_xticks(np.linspace(0, ax.get_xlim()[1], xticks))
        ax.set_xticklabels([':'.join(str(timedelta(minutes=int(d))).split(':')[:2]) for d in np.linspace(0, ax.get_xlim()[1] * duration_iteration, xticks, dtype=int)],
                           rotation=45)
        ax.minorticks_off()

        ax.set_ylabel(i)
        ax2: Axes
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

        # ax2.set_ylabel(ax.get_ylabel())
        ax.yaxis.tick_right()
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax.set_xlabel('duration (h:m)')



        # ax.set_xticks(np.linspace(0, 40, 6))
        # # ax.set_xticklabels(np.linspace(0, 50 * it_lenght[app[_i].lower()]/60, 6))
        # ax.set_xticklabels(np.linspace(0, 50, 6))
        ax.grid(b=True, axis='y', which='major', linestyle='-', alpha=0.5)

    axes[0].set_title(title, loc='left')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, loc='upper center', fontsize='small', ncol=3,
                   bbox_to_anchor=(0.6, 1.4))
    axes[1].set_ylabel('cumulative cost ($)\n' + axes[1].get_ylabel())
    axes[1].get_legend().remove()
    if len(axes) > 2:
        axes[2].get_legend().remove()

    max_x = max([ax.get_xlim()[1] for ax in axes])
    [ax.set_xlim(0, max_x) for ax in axes]


if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 300
    # name = 'trace-quarkus-2021-08-31T18 03 04'  # ICSE (50, 100, 200)
    name = 'trace-quarkus-2021-09-14T19 46 43'  # more initial memory

    name_tuned = 'trace-quarkus-2021-09-22T17 16 44'
    name_tuned = 'trace-quarkus-2021-09-23T14 38 45'
    name_tuned = 'trace-daytrader-2021-09-22T02 42 28' # JSF JSP
    name_tuned = 'trace-acmeair-2021-09-14T19 46 28'
    # name_tuned = 'trace-daytrader-2021-09-15T23 26 02'
    name_non_tuned = 'trace-quarkus2-2021-09-22T17 16 18'
    name_non_tuned = 'trace-quarkus2-2021-09-23T14 38 39'
    name_non_tuned = 'trace-daytrader-2021-09-22T02 42 28' #JSF JSP
    name_non_tuned = 'trace-acmeair-2021-09-14T19 46 28'
    # name_non_tuned = 'trace-daytrader-2021-09-15T23 26 02'


    # title, iteration_duration, simulated_non_tuning = 'Daytrader', 20, True
    title, iteration_duration, simulated_non_tuning = 'AcmeAir', 10, True
    # title, iteration_duration, simulated_non_tuning = 'QHD', 5, False

    df_qhd = load_file_workload(f'resources/{name}.json', iteration_lenght_minutes=iteration_duration)
    df_qhd_tuned = load_file_workload(f'resources/{name_tuned}.json', iteration_lenght_minutes=iteration_duration)
    df_qhd_non_tuned = load_file_workload(f'resources/{name_non_tuned}.json', iteration_lenght_minutes=iteration_duration)

    real_cost(df_qhd_tuned, df_qhd_non_tuned, title=title, duration_iteration=iteration_duration, simulated_non_tuning=simulated_non_tuning)
    plt.show()
