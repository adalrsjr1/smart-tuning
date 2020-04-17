from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.optimizers import Adam, RMSprop

import os

import numpy as np
from hyperopt import hp, STATUS_OK
import matplotlib.pyplot as plt

from common.hyperoptimization import HyperOpt

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


def hyperopt(max_iterations, raw_data, features, max_steps):

    def objective(args):
        print(args)
        _batch = int(args['batch'])
        _epochs = int(args['epochs'])
        _dropout = args['dropout']
        _steps = int(args['steps'])
        _n_layers = int(args['n_layers'])
        _max_neurons = int(args['max_neurons_each_layer'])

        activations = ['tanh', 'softmax', 'sigmoid', 'relu']
        _layer = layer(_n_layers, _max_neurons, activations)
        _model, _history = oracle(raw_data, _steps, features, _layer, _batch, _epochs, 'mean_absolute_error', _dropout,
                                  'adam', verbose=0)

        return {
            'loss': _history.history['loss'][0],
            'status': STATUS_OK,
            'layers': layer,
            'model': _model,
            'history': _history
        }

    def layer(lenght, max_neurons, activations):
        layers = {}
        while len(layers) < lenght:
            layers[np.random.randint(max_neurons)] = np.random.choice(activations)

        return layers

    # define a search space
    space = {
        'n_layers': hp.quniform('n_layers', 1, 10, 1),
        'max_neurons_each_layer': hp.quniform('max_neurons_each_layer', 1, 100, 1),
        'batch': hp.choice('batch', [len(raw_data), 1, 32, 64, 128]),
        'epochs': hp.quniform('epochs', 100, 1000, 1),
        'dropout': hp.uniform('dropout', 0, 0.5),
        'steps': hp.quniform('steps', 1, max_steps, 1)
    }

    opt = HyperOpt(objective, space)

    return opt.optimize(max_iterations)

def add_layers(model, layers, dropout):
    assert len(layers.items()) > 0

    print('creating model')
    for i, t in enumerate(layers.items()):
        print('adding new layer >>>', end=' ')
        if i < len(layers)-1:
            model.add(LSTM(t[0], activation=t[1], return_sequences=True))
            print(f'(n_neurons={t[0]}, activation={t[1]}, dropout={dropout}, return_sequences=True)')
        else:
            model.add(LSTM(t[0], activation=t[1]))
            print(f'(n_neurons={t[0]}, activation={t[1]}), dropout={dropout}')
        if dropout > 0.0:
            model.add(Dropout(dropout))
    print()

def init_model():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    return Sequential()

def set_network_architecture(model, layers, features, output_activation='softmax', dropout=0.0):
    # activation = ['sigmoid', 'relu', 'tanh', 'elu', 'softmax', 'selu', 'softplus', 'softsign', 'hard_sigmoid', 'exponential', 'linear']
    # _model.add(LSTM(steps, activation='tanh'))
    add_layers(model, layers, dropout)
    model.add(Dense(features, activation=output_activation))  # activation
    model.add(Dropout(dropout))

    return model

def encode_input(raw_data, features, steps):
    if not raw_data:
        return None, None
    encoded_raw_seq = to_categorical(raw_data, num_classes=features)
    # number of entries to forecast next
    X, y = split_sequence(encoded_raw_seq, steps)
    X = X.reshape(X.shape[0], X.shape[1], features)

    return  X, y


def oracle(raw_data, steps, features, layers={50: 'tanh'}, batch=32, epochs=100, loss='mean_squared_error',
           dropout=0.0, optimizer='adam', verbose=0, output_activator='tanh', validation_data=None):
    _model = init_model()

    _model = set_network_architecture(_model, layers, features, output_activation=output_activator, dropout=dropout)

    # optimizer = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam', ]
    #
    # loss = [
    # 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
    # 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'huber_loss',
    # 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy',
    # 'kullback_leibler_divergence', 'poisson', 'cosine_proximity', ',is_categorical_crossentropy']
    testX, testy = encode_input(validation_data, features, steps)
    _model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    X, y = encode_input(raw_data, features, steps)

    history = _model.fit(X, y, batch_size=batch, epochs=epochs, verbose=verbose, validation_data=(testX, testy), use_multiprocessing=True)

    # for real time fitting
    # history = None
    # for i in range(epochs):
    #     history = _model.fit(X, y, epochs=1, batch_size=batch, verbose=verbose, shuffle=False)
    #     _model.reset_states()

    return _model, history

def optimization(raw_data, features, steps):
    if hyperopt:
        # {'batch': 31.0, 'dropout': 0.42191428621348437, 'epochs': 178.0, 'n_neurons': 81.0} Error: 41%
        trial, optimal_config = hyperopt(100, raw_data, features, steps)
        print(optimal_config)
        result = trial['result']
        model = result['model']
        history = result['history']
        return model, history

def training(raw_data, features, steps, layers={50: 'tanh'}, batch=32, epochs=100, loss='mean_squared_error',
           dropout=0.0, optimizer='adam', verbose=0, output_activator='tanh', validation_data=None):

    # error: 40%
    return oracle(raw_data, steps, features, verbose=verbose, layers=layers, batch=batch, epochs=epochs, loss=loss,
           dropout=dropout, optimizer=optimizer, output_activator=output_activator, validation_data=validation_data)

def pattern_generator(length, features, seed=0, mutation=0.0):
    np.random.seed(seed)
    data = np.random.randint(features, size=length, dtype=int)
    np.random.seed(None)
    for i in range(length):
        if np.random.rand() < mutation:
            data[i] = np.random.randint(features, dtype=int)

    return data

def data_generator(steps, features, repetitions, seed=0, mutation=0.0):
    raw_seq = []
    # n days
    for i in range(repetitions):
        # one day
        serie = pattern_generator(steps, features, seed=seed, mutation=mutation)
        if i == 0:
            print(f'pattern: {serie}')
        [raw_seq.append(k) for k in serie]
    return raw_seq

def update_model(model, steps, features):
    pass

def experiment_oracle(steps=12, features=4, total=364, training_size=28, layers={30: 'tanh', 7: 'tanh', 24: 'tanh'},
                      dropout=0.1, output_activator='tanh', loss='categorical_crossentropy', optimizer='adam',
                      mutation=0.3, verbose=0, seed=0):

    print(f'experiment setup: steps({steps}), features({features}), total({total}), training_size({training_size}), '
          f'layers({layers}), dropout({dropout}), output_activator({output_activator}), loss({loss}), seed({seed}), '
          f'optimizer({optimizer}), mutation({mutation})')

    raw_seq = data_generator(steps, features, total, seed=seed, mutation=mutation)
    training_seq = raw_seq[:steps*training_size]
    testing_seq = raw_seq[steps*(training_size-1):]

    model, history = training(training_seq, features, steps, epochs=200, dropout=dropout, layers=layers,
                              loss=loss, output_activator=output_activator, verbose=verbose, optimizer=optimizer,
                              validation_data=testing_seq)

    print(model.summary())

    X_train, y_train = encode_input(training_seq, features, steps)
    X_test, y_test = encode_input(testing_seq, features, steps)

    # plot charts
    # evaluate(model, history, X_train, y_train, X_test, y_test)

    new_data_points = []
    # to_predict = input_data.reshape(total - training_size+1, steps, features)

    count = 0
    _countp = 0
    _countn = 0
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)
    interval = 0
    for x, _y in zip(X_test, y_test):
        interval += 1
        yhat = model.predict(x.reshape(1, steps, features), verbose=verbose)

        # print(np.argmax(yhat), np.argmax(_y))
        if np.argmax(yhat) == np.argmax(_y):
            _countp += 1
            _countn += 1
            count += 1

        if interval % steps == 0:
            # print('>>> ', _countp/interval, _countn/interval)
            _countn, _countp = (0, 0)
            interval = 0

        # new value comes up to be evaluateed
        # new_data_points.append(np.argmax(yhat))

        # update model with new batch
        # if len(new_data_points) == steps:
        #     testing_seq = testing_seq[-steps:]
        #     encoded_raw_seq = to_categorical(np.append(testing_seq, new_data_points), features)
        #     X, y = split_sequence(encoded_raw_seq, steps)
        #     X = X.reshape(X.shape[0], X.shape[1], features)
        #     model.train_on_batch(X, y)
        #     # history = model.fit(X, y, batch_size=1, epochs=1, verbose=verbose, use_multiprocessing=True)
        #     new_data_points = []
        #     # previous_data = np.array(raw_seq[-steps:])




    return count/len(y_test)

def evaluate(model, history, trainX, trainy, testX, testy):
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    from timeit import default_timer as timer

    def layers(i):
        list = [
            {30: 'tanh', 7: 'tanh', 24: 'tanh'},
            {30: 'softmax', 7: 'softmax', 24: 'softmax'},
            {50: 'tanh'},
            {50: 'softmax'}
        ]

        return list[i]

    def loss(i):
        list = [
            'categorical_crossentropy',
            'mean_squared_error'
        ]

        return list[i]

    def output_activator(i):
        list = [
            'tanh',
            'softmax'
        ]

        return list[i]

    def dropout(i):
        list = [
            0,
            0.001,
            0.01,
            0.1,
            0.2
        ]

        return list[i]

    def optimizer(i):
        list = [
            # 'sgd',
            'rmsprop',
            # 'adagrad',
            # 'adadelta',
            'adam',
            # 'adamax',
            # 'nadam'
        ]

        return list[i]

    # accuracies = []
    # xs = []
    # # 0.001
    # for lr in [0.001, 0.01, 0.1]:
    #     # 0.9
    #     for b1 in [0.1, 0.9, 0.95, 0.999, 1.8]:
    #         # 0.999
    #         for b2 in [0.1, 0.9, 0.95, 0.999, 1.8]:
    #             # _optimizer = RMSprop(learning_rate=0.001, rho=0.9)
    #             _optimizer = Adam(learning_rate=lr, beta_1=b1, beta_2=b2, amsgrad=False)
    #             # for _optimizer in range(2):
    #             start = timer()
    #             accuracy = experiment_oracle(steps=12, features=4, total=364, training_size=28, layers={20: 'tanh'},
    #                                          dropout=0.01, output_activator='tanh', loss='mean_squared_error', mutation=0.3,
    #                                          optimizer=_optimizer, verbose=0, seed=7)
    #             end = timer()
    #             accuracies.append(accuracy)
    #             xs.append(f'(a:{lr}, b1:{b1}, b2:{b2})')
    #             # xs.append(f'a:{lr}')
    #             print(f'accuracy: {accuracy} --- Time elapsed: {end - start}s\n')
    #
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.set_xlabel('n_neurons')
    # ax.set_ylabel('accuracy')
    # ax.set_title('learning rate')
    #
    # ax.scatter([i for i, _ in enumerate(xs)], accuracies)
    # # ax.plot(range(4, 56, 4), accuracies[0])
    # # ax.plot(range(4, 56, 4), accuracies[1], label=optimizer(1))
    # # plt.xticks([i for i, _ in enumerate(xs)])
    # ax.set_xticks([i for i, _ in enumerate(xs)])
    # ax.set_xticklabels(xs, rotation=90)
    # # plt.yticks(np.arange(0, 1.1, 0.1, ))
    # # plt.legend()
    # plt.show()

    # for _layer in range(4):
    #     for _loss in range(2):
    #         for _output_activator in range(2):
    #             for _dropout in range(5):
    #
    #                 start = timer()
    #                 accuracy = experiment_oracle(steps=12, features=4, total=364, training_size=28, layers=layers(_layer),
    #                                   dropout=dropout(_dropout), output_activator=output_activator(_output_activator),
    #                                              loss=loss(_loss), seed=7, mutation=0.3, verbose=0)
    #                 end = timer()
    #                 print(f'accuracy: {accuracy} --- Time elapsed: {end - start}s\n')

    # rmsp(lr: 0.1) = 0.63
    # adam(lr: 0.001, b1: 0.9, b2: 0.95) = 0.68
    # n_neurons = 20
    # optimizer: [rmsprop, adam]
    # experiment setup: steps(12), features(4), total(364), training_size(28), layers({50: 'tanh'}), dropout(0.01), output_activator(tanh), loss(mean_squared_error), seed(7), mutation(0.3)
    # pattern: [3 0 1 2 3 1 3 3 2 3 2 3]
    # accuracy: 0.654265873015873 --- Time elapsed: 64.83883032299946s

    # experiment setup: steps(12), features(4), total(364), training_size(28), layers({50: 'tanh'}), dropout(0.2), output_activator(tanh), loss(mean_squared_error), seed(7), mutation(0.3)
    # pattern: [3 1 1 2 3 3 3 3 0 1 1 3]
    # accuracy: 0.642609126984127 --- Time elapsed: 63.73357780100014s

    # experiment setup: steps(12), features(4), total(364), training_size(28), layers({30: 'tanh', 7: 'tanh', 24: 'tanh'}), dropout(0.001), output_activator(softmax), loss(categorical_crossentropy), seed(7), mutation(0.3)
    # pattern: [3 0 1 2 1 3 3 1 0 1 2 3]
    # accuracy: 0.6371527777777778 --- Time elapsed: 140.383026943s

    # experiment setup: steps(12), features(4), total(364), training_size(28), layers({50: 'tanh'}), dropout(0.001), output_activator(softmax), loss(mean_squared_error), seed(7), mutation(0.3)
    # pattern: [3 0 1 2 1 3 2 3 0 0 2 3]
    # accuracy: 0.6321924603174603 --- Time elapsed: 66.1867132740008s

    # experiment setup: steps(12), features(4), total(364), training_size(28), layers({30: 'tanh', 7: 'tanh', 24: 'tanh'}), dropout(0), output_activator(softmax), loss(mean_squared_error), seed(7), mutation(0.3)
    # pattern: [2 0 1 3 3 3 3 3 0 1 3 3]
    # accuracy: 0.6326884920634921 --- Time elapsed: 143.42251128299995s

    # experiment setup: steps(12), features(4), total(364), training_size(28), layers({50: 'tanh'}), dropout(0), output_activator(softmax), loss(mean_squared_error), seed(7), mutation(0.3)
    # pattern: [3 0 1 2 3 3 3 3 0 1 2 3]
    # accuracy: 0.6393849206349206 --- Time elapsed: 61.985655982000026s

    # experiment setup: steps(12), features(4), total(364), training_size(28), layers({50: 'tanh'}), dropout(0.2), output_activator(softmax), loss(mean_squared_error), seed(7), mutation(0.3)
    # pattern: [3 0 1 2 3 3 3 0 0 1 2 3]
    # accuracy: 0.6329365079365079 --- Time elapsed: 65.0734399869998s


    mutation = 0
    mutation_list = []
    accuracy_list = []

    seeds = []
    steps = 1
    # for i in range(10):
    #     seed = np.random.randint(1000)
    # while mutation <= 1.05:
    _optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.95, amsgrad=False)

    start = timer()
    accuracy = experiment_oracle(steps=12, features=4, total=364, training_size=28, layers={20: 'tanh'},
                                 dropout=0.01, output_activator='tanh', loss='mean_squared_error', mutation=0.3,
                                 optimizer=_optimizer, verbose=0, seed=7)
    # accuracy = seed * mutation
    end = timer()
    print(f'mutation: {mutation} accuracy: {accuracy} --- Time elapsed: {end - start}s\n')
    accuracy_list.append(accuracy)

    fig = plt.figure()
    ax = plt.axes()
    plt.xticks(range(1, 13))
    ax.set_xlabel('steps')
    ax.set_ylabel('model accuracy')
    ax.set_title('accuracy per #steps')

    ax.plot(accuracy_list)

    plt.legend()
    plt.show()


