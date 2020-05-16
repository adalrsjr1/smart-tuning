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

def plot(filepath, title):
    df:pd.DataFrame = load_rawdata(filepath)

    # df.quantile(.x)
    # prod_avg = df['prod. pod'].quantile(.50)
    # tuni_avg = df['train. pod'].quantile(.50)
    prod_avg = df['prod. pod'].mean()
    tuni_avg = df['train. pod'].mean()


    df['expected'] = [1000] * len(df)
    df['average with default config.'] = [int(1677/2)] * len(df)
    df['p99 prod.'] = [prod_avg] * len(df)
    df['p99 train.'] = [tuni_avg] * len(df)

    ax:plt.Axes = df.plot(drawstyle="steps", linewidth=0.7, style=['-', '-', '--', '--', '--', '--'], rot=0)
    ax.legend(frameon=False)

    ax.set_xlabel('iterations')
    ax.xaxis.set_major_locator(plt.IndexLocator(base=1, offset=1))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_tick_formatter))

    ax.set_ylabel('throuhgput (req/s)')
    ax.set_yticks(np.arange(0, 2500, 200))

    print(prod_avg, tuni_avg, prod_avg-tuni_avg)

    avg_diff = abs(prod_avg - tuni_avg)

    if avg_diff > 100:
        yticks = sorted([1677, 2000, int(prod_avg), int(tuni_avg)])
    else:
        yticks = sorted([1677, 2000, int((prod_avg + tuni_avg)/2)])

    ax.set_yticks(yticks, minor=True)
    labels = [str(tick) for tick in yticks]
    ax.set_yticklabels(labels, minor=True)

    ax.set_title(title)
    plt.show()

if __name__ == '__main__':
    plot('volume/mongo/20200515-063218/mongo_metrics.json', '')
