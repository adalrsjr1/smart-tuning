from pandas import DataFrame, Series
import pandas
from mock import mock
from sklearn.metrics import mean_squared_error
import math
from matplotlib import pyplot as plt
from datetime import datetime


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pandas.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
# making stationary
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


if __name__ == '__main__':
    def parser(x):
        return datetime.strptime('20' + x, '%Y-%m')


    values = mock.periodic_time_series(10, 12, mutation_probability=0)

    series = DataFrame([[i, v] for i, v in enumerate(values)])
    
    print(series.head())
    # transform to be stationary
    differenced = difference(series, 1)
    print(differenced.head())
    # invert transform
    inverted = list()
    for i in range(len(differenced)):
        value = inverse_difference(series, differenced[i], len(series) - i)
        inverted.append(value)
    inverted = Series(inverted)
    print(inverted.head())

