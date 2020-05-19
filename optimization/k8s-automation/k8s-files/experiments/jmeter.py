import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates

import datetime

def load_rawdata(filepath):
    df = pd.read_csv(filepath, na_values='-', usecols=[0, 1], names=['time', 'measured'])
    df.fillna(0, inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('time').sort_index()

    df = df.drop(df.index[0:3])
    return df

last_value = 0
def x_tick_formatter(value, tick_number):
    global last_value
    if last_value == 0:
        last_value = datetime.datetime.fromtimestamp(value)
        return '0'

    current_value = datetime.datetime.fromtimestamp(value)
    label = current_value - last_value

    return str(label).split(' ')[-1]

def plot(ax, filepath, title, timestep, interval, expected_avg):
    df:pd.DataFrame = load_rawdata(filepath)
    average = float(df.rolling('1ms').count().sum() / interval)
    df = df.rolling('1ms').count().groupby(pd.Grouper(freq=f'{timestep}s')).sum() / timestep

    df['expected'] = [expected_avg] * len(df)
    # df['avg w/default config.\n        wo/SmartTuning'] = [1677] * len(df)
    df['avg'] = [average] * len(df)

    ax:plt.Axes = df.plot(ax=ax, drawstyle="steps", linewidth=0.7, style=['b-', 'k--', 'r--', 'g--'], rot=45, title=title)
    ax.legend(frameon=False)

    ax.set_xlabel('time elapsed (h:m:s)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(21))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_tick_formatter))

    ax.set_ylabel('throuhgput (req/s)')

    # yticks = sorted([1677, expected_avg, int(average)])
    # ax.set_yticks(yticks, minor=True)
    # labels = [str(tick) for tick in yticks]
    # ax.set_yticklabels(labels, minor=True)

    # plt.show()
    ax.margins(x=0)
    return ax

if __name__ == '__main__':
    ##
    # for multiple plots reger to: https://stackoverflow.com/questions/38989178/pandas-plot-combine-two-plots
    ##
    plot(ax=None, title=f'throughput measured at client',
         filepath='volume/jmeter/20200515-150051/raw_data_20200515150051.jtl',
         timestep=600  ,
         interval=4*3600,
         expected_avg=2000)
