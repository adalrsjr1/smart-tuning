from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers import Dropout

import numpy as np

from mock.mock import time_series

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def hyperopt(max_iterations, raw_data, features):
    from common.hyperoptimization import HyperOpt
    from hyperopt import hp, STATUS_OK

    def objective(args):
        n_neurons = int(args['n_neurons'])
        _batch = int(args['batch'])
        _epochs = int(args['epochs'])
        _dropout = args['dropout']

        _model, _history = oracle(n_neurons, _dropout, 'adam', 'mean_absolute_error', _batch, _epochs, raw_data, steps,
                                  features, verbose=0)

        return {
            'loss': _history.history['loss'][0],
            'status': STATUS_OK,
            'model': _model,
            'history': _history
        }

    # define a search space
    space = {
        'n_neurons': hp.quniform('n_neurons', 10, 100, 1),
        'batch': hp.quniform('batch', 1, 64, 1),
        'epochs': hp.quniform('epochs', 1, 300, 1),
        'dropout': hp.uniform('dropout', 0, 0.5)
    }

    opt = HyperOpt(objective, space)

    return opt.optimize(max_iterations)

def oracle(n_neurons, dropout, optimizer, loss, batch, epochs, raw_data, steps, features, verbose=0):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _model = Sequential()

    # activation = ['sigmoid', 'relu', 'tanh', 'elu', 'softmax', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential', 'linear']
    _model.add(LSTM(n_neurons))
    _model.add(Dense(features))  # activation
    _model.add(Dropout(dropout))

    # optimizer = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam', ]
    #
    # loss = [
    # 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
    # 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'huber_loss',
    # 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy',
    # 'kullback_leibler_divergence', 'poisson', 'cosine_proximity', ',is_categorical_crossentropy']
    _model.compile(optimizer=optimizer, loss=loss)

    encoded_raw_seq = to_categorical(raw_data, num_classes=features)
    # number of entries to forecast next
    n_steps = steps
    X, y = split_sequence(encoded_raw_seq, n_steps)

    n_features = features
    X = X.reshape(X.shape[0], X.shape[1], n_features)

    history = _model.fit(X, y, batch_size=batch, epochs=epochs, verbose=verbose)

    return _model, history

def realtime_oracle(n_neurons, dropout, optimizer, loss, batch, epochs, raw_data, steps, features, verbose=0):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _model = Sequential()

    # activation = ['sigmoid', 'relu', 'tanh', 'elu', 'softmax', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential', 'linear']
    _model.add(LSTM(n_neurons))
    _model.add(Dense(features))  # activation
    _model.add(Dropout(dropout))

    # optimizer = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam', ]
    #
    # loss = [
    # 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
    # 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'huber_loss',
    # 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy',
    # 'kullback_leibler_divergence', 'poisson', 'cosine_proximity', ',is_categorical_crossentropy']
    _model.compile(optimizer=optimizer, loss=loss)

    encoded_raw_seq = to_categorical(raw_data)
    # number of entries to forecast next
    n_steps = steps
    X, y = split_sequence(encoded_raw_seq, n_steps)

    n_features = features
    X = X.reshape(X.shape[0], X.shape[1], n_features)

    history = None
    for i in range(epochs):
        history = _model.fit(X, y, epochs=1, batch_size=batch, verbose=verbose, shuffle=False)
        _model.reset_states()

    return _model, history

def checkvalue(n, t, p=0):
    v= getvalue(t, p)
    return v == n, v

def getvalue(t, p=0):
    i = t % 10

    if p != 0:
        if t % 7 == 0:
            return 3
        if t % 13 == 0:
            return 2
        if t % 17 == 0:
            return 1
        if t % 19 == 0:
            return 0
    return time_series(10, 0)[i]


def experiment_oracle():
    raw_seq = []
    # n days
    for i in range(10):
        # one day
        serie = time_series(10, 0.0)
        if i == 0:
            print(serie)
        # [raw_seq.append(k) for k in serie]
        [raw_seq.append(k%4) for k in range(10)]

    steps = 10
    features = 4

    # trial, optimal_config = hyperopt(200, raw_seq, features)
    # print(optimal_config)
    # result = trial['result']
    # {'batch': 31.0, 'dropout': 0.42191428621348437, 'epochs': 178.0, 'n_neurons': 81.0} Error: 41%
    # model = result['model']
    # history = result['history']

    # error: 40%
    model, history = oracle(50, 0.0, 'adam', 'mean_squared_error', 32, 200, raw_seq, steps, features, verbose=1)

    previous_data = np.array(raw_seq[-steps:])
    count = 0
    length = 3000
    new_data_points = []
    for i in range(length):
        to_predict = to_categorical(previous_data, features)
        to_predict = to_predict.reshape(1, steps, features)
        yhat = model.predict(to_predict)

        # print(np.argmax(yhat), getvalue(i))

        # new value comes up to be evaluateed
        # yhat foreseen, v real
        foreseen = np.argmax(yhat)
        isequal, v = checkvalue(foreseen, i, p=0.3)
        if not isequal:
            count += 1
            foreseen = v

        new_data_points.append(foreseen)

        previous_data = np.append(previous_data[1:], foreseen)
        # update model with new batch
        if len(new_data_points) == steps:
            raw_seq = raw_seq[-steps:]
            encoded_raw_seq = to_categorical(np.append(raw_seq, new_data_points), features)
            X, y = split_sequence(encoded_raw_seq, steps)
            X = X.reshape(X.shape[0], X.shape[1], features)
            history = model.fit(X, y, batch_size=32, epochs=1, verbose=0)
            new_data_points = []
            previous_data = np.array(raw_seq[-steps:])

    print(f'error: {count/len(range(length))}')



    # # start with n=10 initial values
    # test_y = raw_seq
    # results = []
    #
    # count = 0
    # for i in range(1, 101):
    #     net_input = test_y[steps*i:]
    #     net_input = to_categorical(net_input, num_classes=features)
    #
    #     X, y = split_sequence(net_input, steps)
    #     X = X.reshape(X.shape[0], X.shape[1], features)
    #
    #     yhat = model.predict(X, verbose=0)
    #
    #     if not checkvalue(np.argmax(yhat), 100+i):
    #         count += 1
    #
    #     # print(yhat, np.argmax(yhat))
    #
    #     # print((np.argmax(yhat), checkvalue(np.argmax(yhat), i)), end=' ')
    #     test_y = np.append(test_y, np.argmax(yhat))

    # input = to_categorical(raw_seq)[-steps:]
    # count = 0
    # interval = range(100)
    # for i in interval:
    #     x_input = np.array(input)
    #     x_input = x_input.reshape((1, steps, features))
    #     yhat = model.predict(x_input, verbose=0)
    #     if not checkvalue(np.argmax(yhat), i):
    #         count += 1
    #     print((np.argmax(yhat), checkvalue(np.argmax(yhat), i)), end=' ')
    #     input = np.append(input, yhat)
    #     input = input.reshape((steps + 1, features))
    #     input = input[1:]
    # print(f'error: {count / len(interval)}')


def experiment_realtime_oracle():
    raw_seq = []
    for i in range(0, 1):
        serie = time_series(10, 0.)
        [raw_seq.append(k) for k in serie]

    steps = max(raw_seq)
    features = max(raw_seq) + 1

    # trial, optimal_config = hyperopt(200, raw_seq, features)
    # print(optimal_config)
    # result = trial['result']
    # {'batch': 31.0, 'dropout': 0.42191428621348437, 'epochs': 178.0, 'n_neurons': 81.0} Error: 41%
    # model = result['model']
    # history = result['history']

    # error: 40%
    model, history = oracle(50, 0.0, 'adam', 'mean_squared_error', 32, 200, raw_seq, steps, features, verbose=1)

    # X, y = split_sequence(input, steps)
    # model.fit(X, y, batch_size=32, epochs=1, verbose=1)

    input = to_categorical(raw_seq)[-steps:]
    count = 0
    interval = range(100)
    for i in interval:
        x_input = np.array(input)
        x_input = x_input.reshape((1, steps, features))
        yhat = model.predict(x_input, verbose=0)

        if not checkvalue(np.argmax(yhat), i):
            count += 1
        print((np.argmax(yhat), checkvalue(np.argmax(yhat), i)), end=' ')

        input = np.append(input, yhat)

        input = input.reshape((steps + 1, features))
        input = input[1:]



    print(f'error: {count / len(interval)}')

if __name__ == '__main__':
    experiment_oracle()