import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import hashlib
import json

def load_rawdata(filepath):
    data = {'date': [], 'prod. pod': [],
            # 'train. pod': [],
            'config': []}
    first = 0
    with open(filepath) as f:
        for doc in f:
            doc_parsed = json.loads(doc)
            data['prod. pod'].append(float(doc_parsed['prod_workload']['metric']))
            # data['train. pod'].append(float(doc_parsed['training_workload']['metric']))
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

    return str(label)

def plot(filepath, title, timestep):
    df:pd.DataFrame = load_rawdata(filepath)
    print(df)

    ax:plt.Axes = df.plot(drawstyle='steps', linewidth=0.7, style=['k-', '-', '--', '--', '--', '--'], rot=0)
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

    xs, ys = [], []
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

        ax.step(xs, ys, drawstyle='steps', label='config: ' + item[0][:6])
        ax.legend()



    #     if xs and ys:
    #         xs.append(p[0][0])
    #         ys.append(p[0][1])
    #         ax.scatter(xs, ys)
    #         # ax.plot(xs, ys)
    #         xs, ys = [], []
    #     else:
    #         xs, ys = [], []
    #
    #     for x, y in p:
    #         xs.append(x)
    #         ys.append(y)
    #
    # ax.scatter(xs, ys)
    # # ax.plot(xs, ys)


        # for _p in p:
            # ax.step(xs, ys, drawstyle='steps-post')
            # ax.scatter(_p[0], _p[1], marker='o')


    ax.set_yticks(np.arange(0, 2500, 200))
    ax.set_title(title)
    plt.show()

if __name__ == '__main__':
    plot('volume/mongo/20200515-190326/mongo_workloads.json', '', timestep=600)
