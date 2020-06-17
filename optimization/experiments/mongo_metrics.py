import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json

def load_rawdata(filepath):
    data = {'prod. pod': [], 'train. pod': []}
    with open(filepath) as f:
        for doc in f:
            doc_parsed = json.loads(doc)
            data['prod. pod'].append(float(doc_parsed['prod_metric']))
            data['train. pod'].append(float(doc_parsed['tuning_metric']))

    return pd.DataFrame(data)

def x_tick_formatter(value, tick_number):
    return str(int(value))

def plot(ax, filepath, title, expected_avg):
    df:pd.DataFrame = load_rawdata(filepath)

    # df.quantile(.x)
    # prod_avg = df['prod. pod'].quantile(1)
    # tuni_avg = df['train. pod'].quantile(1)
    prod_avg = df['prod. pod'].mean()
    tuni_avg = df['train. pod'].mean()


    df['expected'] = [expected_avg] * len(df)
    # df['average with default config.'] = [int(1677/2)] * len(df)
    df['avg prod.'] = [prod_avg] * len(df)
    df['avg train.'] = [tuni_avg] * len(df)

    print(df)

    ax:plt.Axes = df.plot(ax=ax, drawstyle="steps-post", linewidth=0.7, style=['-', '-', '--', '--', '--', '--'], rot=0, title=title)
    ax.legend(frameon=False)
    ax.set_ylim(0, 100+max(df['prod. pod'].max(), df['train. pod'].max()))
    ax.set_xlabel('iterations')
    ax.xaxis.set_major_locator(plt.IndexLocator(base=1, offset=1))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_tick_formatter))

    ax.set_ylabel('throuhgput (req/s)')
    # ax.set_yticks(np.arange(0, 2500/2, 200))

    # print(prod_avg, tuni_avg, prod_avg-tuni_avg)

    avg_diff = abs(prod_avg - tuni_avg)

    # if avg_diff > 100:
    #     yticks = sorted([expected_avg, int(prod_avg), int(tuni_avg)])
    # else:
    #     yticks = sorted([expected_avg, int((prod_avg + tuni_avg)/2)])
    #
    # ax.set_yticks(yticks, minor=True)
    # labels = [str(tick) for tick in yticks]
    # ax.set_yticklabels(labels, minor=True)

    ax.margins(x=0)
    # plt.show()
    return ax

if __name__ == '__main__':
    plot(None, 'volume/mongo/20200531-225554/mongo_metrics.json', '', expected_avg=1000)
    plt.show()
