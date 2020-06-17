import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import dates

import hashlib
import json

def load_rawdata(filepath):
    data = {'date': [], 'prod. pod': [],
            'train. pod': [],
            'config': []}
    first = 0
    with open(filepath) as f:
        for doc in f:
            doc_parsed = json.loads(doc)
            data['prod. pod'].append(float(doc_parsed['prod_workload']['metric']))
            data['train. pod'].append(float(doc_parsed['training_workload']['metric']))
            data['config'].append(hashlib.md5(str(doc_parsed['prod_workload']['configuration']).encode('utf-8')).hexdigest())

            if first == 0:
                first = int(doc_parsed['prod_workload']['start'])

            data['date'].append(int(doc_parsed['prod_workload']['start']))
            # data['train. pod'].append(float(doc_parsed['tuning_metric']))

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df = df.set_index('date').sort_index()

    return df

last_value = 0
def x_tick_formatter(value, tick_number):
    global last_value

    if last_value == 0:
        last_value = datetime.datetime.fromtimestamp(value * 1000 * 60)
        return '0'

    current_value = datetime.datetime.fromtimestamp(value * 1000 * 60)
    label = current_value - last_value

    return str(label).split(' ')[-1]

last_value = 0
def x_tick_formatter(value, tick_number):
    global last_value
    formatter = dates.DateFormatter('%H:%M:%S')
    if last_value == 0:
        last_value = datetime.strptime(formatter.format_data_short(value), '%H:%M:%S')
        return '0'

    current_value = datetime.strptime(formatter.format_data_short(value), '%H:%M:%S')
    label = current_value - last_value
    return str(label).split(' ')[-1]

def plot(ax, filepath, title, expected_avg):
    df:pd.DataFrame = load_rawdata(filepath)
    ax = df.plot(ax=ax, drawstyle='steps-post', linewidth=0.7, y=['prod. pod'], style=['k-', '-', '--', '--', '--', '--'], rot=45, title=title)
    ax.legend(frameon=False)
    print(df)
    ax.set_ylim(0, 100 + max(df['prod. pod'].max(), df['train. pod'].max()))
    points = {}
    for line in ax.get_lines():
        if line.get_label() != 'prod. pod':
            continue

        for index, point in enumerate(line.get_xydata()):
            config = df['config'][index]
            if config in points:
                points[config].append(point)
            else:
                points[config] = [point]

    items = [item for item in points.items()]
    for i, item in enumerate(items):
        _points = item[1]

        xs, ys = [], []
        for x, y in _points:
            xs.append(x)
            ys.append(y)

        if i+1 < len(items):
            _points = items[i+1][1]
            xs.append(_points[0][0])
            ys.append(_points[0][1])

        print(xs, ys)

        ax.plot(xs, ys, drawstyle='steps-post', label='config: ' + item[0][:6])
        ax.legend(frameon=False)

    ax.xaxis.set_major_locator(plt.MaxNLocator(21))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_tick_formatter))

    # ax.set_yticks(np.arange(0, 2500, 200))
    ax.margins(x=0)
    # plt.show()
    return ax

if __name__ == '__main__':
    plot(None, 'volume/mongo/20200531-225554/mongo_workloads.json', '', expected_avg=2000)
    plt.show()
