import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

    return str(label)

def plot(filepath, title, timestep, interval):
    df:pd.DataFrame = load_rawdata(filepath)

    average = float(df.rolling('1ms').count().sum() / interval)
    df = df.rolling('1ms').count().groupby(pd.Grouper(freq=f'{timestep}s')).sum() / timestep

    df['expected'] = [2000] * len(df)
    df['average with default config.'] = [1677] * len(df)
    df['average with SmartTuning'] = [average] * len(df)

    ax:plt.Axes = df.plot(drawstyle="steps", linewidth=0.7, style=['b-', 'k--', 'r--', 'g--'], rot=45)
    ax.legend(frameon=False)

    ax.set_xlabel('time elapsed (h:m:s)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(timestep / 25))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_tick_formatter))

    ax.set_ylabel('throuhgput (req/s)')

    yticks = sorted([1677, 2000, int(average)])
    ax.set_yticks(yticks, minor=True)
    labels = [str(tick) for tick in yticks]
    ax.set_yticklabels(labels, minor=True)

    ax.set_title(title)
    plt.show()

if __name__ == '__main__':
    ##
    # for multiple plots reger to: https://stackoverflow.com/questions/38989178/pandas-plot-combine-two-plots
    ##
    plot(title=f'throughput measured at client',
         filepath='volume/jmeter/20200515-150051/raw_data_20200515150051.jtl',
         timestep=600,
         interval=4*3600)
