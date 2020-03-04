import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import minmax_scale
from scipy.spatial import distance

from time import time, sleep


def sinusoidal1(t, mean=0, stddev=0):
    return np.abs(
        np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * 2.25 * 2 * np.pi * t) + np.random.normal(mean, stddev, len(t)))


def sinusoidal2(t, mean=0, stddev=0):
    return np.abs(np.sin(2 * np.pi * t))


def sinusoidal3(t, mean=0, stddev=0):
    return np.sin(2 * 2.25 * 2 + np.pi * t)


def sinusoidal_interval(which_interval, n_buckets, mean=0, stddev=0):
    subintervals = np.linspace(0, 2, 5)

    assert len(subintervals) >= which_interval >= 0

    i0 = subintervals[which_interval]
    i1 = subintervals[which_interval + 1]

    t = np.linspace(i0, i1, n_buckets)
    s = sinusoidal1(t, mean, stddev)

    return s


def series(t):
    if t % 7 == 0:
        return 0
    if t % 13 == 1:
        return 1
    if t % 17 == 2:
        return 2
    if t % 19 == 3:
        return 3

    return -1


def long_series(size, mutation_probability=0.0):
    serie = []

    count = 0
    while len(serie) < size:
        v = series(count)
        if v >= 0:
            serie.append(v)
        r = np.random.rand()
        count += 1
        if r < mutation_probability:
            count += int(r * 10) - 1

    return serie


def plot(x, y):
    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")
    plt.plot(x, y)
    plt.show()


def lstm():
    # LSTM for international airline passengers problem with window regression framing
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    from pandas import DataFrame
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error

    import io
    import requests

    # url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    # s = requests.get(url).content

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)

    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    # dataframe = read_csv(io.StringIO(s.decode('utf-8')), usecols=[1], engine='python')
    dataframe = DataFrame(np.array([long_series(12, 0.3) for i in range(10)]).flatten())

    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


if __name__ == '__main__':
    for i in range(10):
        print(long_series(10, 0.3))
