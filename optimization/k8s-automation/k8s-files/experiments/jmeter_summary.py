import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import dates

_time_ = 'time'
_max_ = 'max resp. time. (ms)'
_min_ = 'min resp. time (ms)'
_avg_ = 'avg resp. time (ms)'
_sum_ = 'req/s'
_err_ = 'errors'

def load_rawdata(filepath, label):

    data = {_time_: [], label + ' ' + _sum_: [],label + ' ' +  _avg_: [], label + ' ' + _min_: [], label + ' ' + _max_: [], label + ' ' + _err_: []}
    with open(filepath) as f:
        #o.a.j.r.Summariser
        pattern = '^(?P<time>2020.*) INFO (o\.a\.j\.r\.Summariser: summary [=].+ =\s+)(?P<summary>\d+\.\d+/s)( Avg:\s+)(?P<avg>\d+)( Min:\s+)(?P<min>\d+)( Max:\s+)(?P<max>\d+)( Err:\s+)(?P<err>\d+)'
        for row in f:
            search = re.findall(pattern, row)

            if search:
                search = search[0]
                data[_time_].append(search[0].replace(',','.'))
                data[label + ' ' + _sum_].append(float(search[2].split('/')[0]))
                data[label + ' ' + _avg_].append(int(search[4]))
                data[label + ' ' + _min_].append(int(search[6]))
                data[label + ' ' + _max_].append(int(search[8]))
                data[label + ' ' + _err_].append(int(search[10])/6)

    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f')
    df = df.set_index('time').sort_index()

    return df

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


def plot(ax, filename, title, expected_avg, label):
    df = load_rawdata(filename, label)

    ax = df.plot(ax=ax, linewidth=0.7, y=[label+' '+_sum_, label+' '+_avg_, label+' '+_err_], title=title)

    ax.legend(frameon=False)


    ax.set_xlabel('time elapsed (h:m:s)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(21))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_tick_formatter))
    # date_fmt = '%H:%M:%S'
    # formatter = dates.DateFormatter(date_fmt)
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(x_tick_formatter))
    ax.margins(x=0)
    return ax

if __name__ == '__main__':

    ax = plot(None, 'volume/jmeter/prod/20200522-014432/acmeair.stats.2020052214432',
         '', expected_avg=1000, label='prod')

    ax = plot(ax, 'volume/jmeter/train/20200522-014432/acmeair.stats.2020052214432',
              '', expected_avg=1000, label='train')

    plt.show()