import pandas as pd
import matplotlib.pyplot as plt

def load_rawdata(path):
    df = pd.read_csv(path, na_values='-', usecols=[0,1], names=['time', 'throughput'])
    df.fillna(0, inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('time').sort_index()

    df = df.drop(df.index[0:3])
    return df

if __name__ == '__main__':

    df:pd.DataFrame = load_rawdata('volume/jmeter/20200514-170621/raw_data_20200514170621.jtl')

    interval = 60
    df = df.rolling('1ns').count().groupby(pd.Grouper(freq=f'{interval}s')).sum() / interval
    df['expected'] = [2000 for _ in range(len(df))]

    ax = df.plot(drawstyle="steps", linewidth=0.7, style=['b-', 'r--'])
    plt.show()
