import copy
from datetime import datetime, timedelta
from pprint import pprint
from scipy.spatial import distance
import re
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
        'cfg_data': [],
        'cfg_score': [],
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
        't_cfg_data': [],
        't_cfg_score': [],
        't_cost': [],
        't_cost_req': [],
    }

    def extract_cfg_values(cfg: dict):
        values = []
        for config_map_name, config_map_data in cfg.items():
            for parameter, value in config_map_data.items():
                new_value = value
                if isinstance(value, str):
                    if value.isnumeric():
                        new_value = float(value)
                    else:
                        # new_value = int.from_bytes(bytes('test', encoding='ascii'), 'big')
                        # guarantee only postive values
                        continue
                        new_value = hash(value) & sys.maxsize
                else:
                    new_value = float(value)
                values.append(new_value)

        return values

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
            table['cfg_data'].append(extract_cfg_values(record['production']['curr_config']['data']))
            table['cfg_score'].append(record['production']['curr_config']['stats']['median'] * -1)
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
            table['t_cfg_data'].append(extract_cfg_values(record['training']['curr_config']['data']))
            table['t_cfg_score'].append(record['training']['curr_config']['stats']['median'] * -1)
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


def plot_cfg_dispersion(df: pd.DataFrame, title=''):
    grouped: pd.core.groupby.generic.DataFrameGroupBy = df.groupby('workload')

    fig: Figure
    fig, axes = plt.subplots(nrows=len(grouped),
                             # figsize=(6,6),
                             ncols=1,
                             sharex='all',
                             )
    fig.set_constrained_layout(True)
    # fig.tight_layout()

    for _groups in grouped:
        i, groups = _groups
        df = groups.reset_index(drop=True)

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

        initial_cfg = df.iloc[0]
        df['t_distance'] = df.apply(lambda row: distance.cosine(initial_cfg['cfg_data'], row['t_cfg_data']), axis=1)
        df['distance'] = df.apply(lambda row: distance.cosine(initial_cfg['cfg_data'], row['cfg_data']), axis=1)
        best_cfg = df.query('t_cfg_score == t_cfg_score.max()')
        best_cfg = best_cfg.iloc[-1]

        ax: Axes = workloads[i]
        im = ax.scatter(x=df.index, y=df.t_distance, c=df.t_cfg_score,
                  cmap='rainbow', marker='.', label='training cfg', vmin=df.t_cfg_score.min(), vmax=df.t_cfg_score.max())

        ax.scatter(x=best_cfg.name, y=best_cfg.t_distance, c=best_cfg.t_cfg_score,
                        cmap='rainbow', marker='*', label='best cfg', vmin=df.t_cfg_score.min(), vmax=df.t_cfg_score.max())

        # ax.set_yticks(np.linspace(0, 1, 4))
        # ax.set_yticklabels([f'{y:.2f}' for y in np.linspace(0, 1, 4)])
        ax.set_ylim(0)
        ax.set_xlim(0)

        cbar = fig.colorbar(im, ax=ax, shrink=0.95)
        cbar.set_ticks([df.t_cfg_score.min(), df.t_cfg_score.max()])
        cbar.set_ticklabels(['worst', 'best'])

    axes[0].set_title(title, loc='left')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, loc='upper center', fontsize='small', ncol=3,
                   bbox_to_anchor=(0.6, 1.4))
    # axes[1].get_legend().remove()
    # if len(axes) > 2:
    #     axes[2].get_legend().remove()
    axes[1].set_ylabel('cosine distance from initial cfg')
    axes[-1].set_xticks(np.arange(0, 55, 5))
    axes[-1].set_xlabel('iteration')



if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 300
    # name = 'trace-quarkus-2021-08-31T18 03 04'  # ICSE (50, 100, 200)
    # name = 'trace-quarkus-2021-09-14T19 46 43'  # more initial memory

    name = 'trace-quarkus-2021-09-14T19 46 43'  # Azure
    # name = 'trace-quarkus-2021-09-23T14 38 45' # Trinity
    # name = 'trace-daytrader-2021-09-22T02 42 28' # JSF JSP
    # name = 'trace-acmeair-2021-09-14T19 46 28'
    # name = 'trace-daytrader-2021-09-15T23 26 02'
    # name_non_tuned = 'trace-quarkus-2021-09-14T19 46 43'  # Azure
    # name_non_tuned = 'trace-quarkus2-2021-09-23T14 38 39' # Trinity
    # name_non_tuned = 'trace-quarkus-2021-09-23T14 38 45' # Trinity
    # name_non_tuned = 'trace-daytrader-2021-09-22T02 42 28' #JSF JSP
    # name_non_tuned = 'trace-acmeair-2021-09-14T19 46 28'
    # name_non_tuned = 'trace-daytrader-2021-09-15T23 26 02'

    # title, iteration_duration, simulated_non_tuning = 'Daytrader', 20, True
    # title, iteration_duration, simulated_non_tuning = 'AcmeAir', 10, True
    title, iteration_duration, simulated_non_tuning = 'QHD', 5, True

    df = load_file_workload(f'resources/{name}.json', iteration_lenght_minutes=iteration_duration)
    plot_cfg_dispersion(df, title=title)
    # df_qhd_non_tuned = load_file_workload(f'resources/{name_non_tuned}.json',
    #                                       iteration_lenght_minutes=iteration_duration)

    plt.show()
