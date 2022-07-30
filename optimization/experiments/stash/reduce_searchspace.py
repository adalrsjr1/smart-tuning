import pandas as pd
import hashlib
import kubernetes
import os, sys, time
import json

import sampler

import keras
from keras import Sequential
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import  matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import random as python_random

def reset_seeds():
   np.random.seed(123)
   python_random.seed(123)
   tf.random.set_seed(1234)

reset_seeds()

SEED=0
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.execv(sys.executable, [sys.executable] + sys.argv)

def load_configs(filename:str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df.fillna(0)

    configs_names = [name for name in df.columns if name.startswith('last_config')]
    tconfigs_names = [name for name in df.columns if name.startswith('config_to_eval')]
    reduced_table = df[configs_names]
    treduced_table = df[tconfigs_names]
    names_parsed = [name.split('.')[1:] for name in configs_names]


    configs = []
    tconfigs = []
    configs_name = []
    tconfigs_name = []

    for index, row in reduced_table.iterrows():
        unique = hashlib.md5(bytes(str(tuple(row.values[1:])), 'ascii')).hexdigest()
        tunique = hashlib.md5(bytes(str(tuple(treduced_table.iloc[index].values[1:])), 'ascii')).hexdigest()
        config = {names[0]: {} for names in names_parsed if len(names) > 0}
        tconfig = {names[0]: {} for names in names_parsed if len(names) > 0}

        for key, value in row.items():
            tag = key.split('.')[1:]
            if len(tag) > 0:
                config[tag[0]][tag[1]] = value

        for key, value in treduced_table.iloc[index].items():
            tag = key.split('.')[1:]
            if len(tag) > 0:
                tconfig[tag[0]][tag[1]] = value


        configs.append(config)
        tconfigs.append(tconfig)
        configs_name.append(unique[:3])
        tconfigs_name.append(tunique[:3])

    # df= pd.DataFrame({'names':configs_name,  'configs':pd.DataFrame.from_dict(configs), 'values':df['production_metric.objective'].values * -1,
    #                   'tnames':tconfigs_name, 'tconfigs':pd.DataFrame.from_dict(tconfigs), 'tvalues':df['training_metric.objective'].values * -1})
    prod = pd.json_normalize(configs, sep='_').assign(values=df['production_metric.objective'].values * -1)
    train = pd.json_normalize(configs, sep='_').assign(values=df['training_metric.objective'].values * -1)

    columns = prod.columns
    cpu_binarizer = LabelBinarizer().fit(train["daytrader-service_cpu"])
    mem_binarizer = LabelBinarizer().fit(train["daytrader-service_memory"])

    prod['daytrader-service_cpu'] = cpu_binarizer.transform(prod['daytrader-service_cpu'])
    prod['daytrader-service_memory'] = mem_binarizer.transform(prod['daytrader-service_memory'])

    train['daytrader-service_cpu'] = cpu_binarizer.transform(train['daytrader-service_cpu'])
    train['daytrader-service_memory'] = mem_binarizer.transform(train['daytrader-service_memory'])

    for column_name in columns:
        if 'cpu' in column_name or 'memory' in column_name:
            continue

        prod[column_name] /= 1000#prod[column_name].max()
        train[column_name] /= 1000#train[column_name].max()

    return prod, train



def learning(df: pd.DataFrame, tdf: pd.DataFrame) -> Sequential:
    columns = list(df.columns)
    columns.remove('values')

    x = tdf[columns].values
    y = tdf['values'].to_numpy()
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    x_ = df[columns].values
    y_ = df['values'].to_numpy()
    x_ = np.expand_dims(x_, axis=1)
    y_ = np.expand_dims(y_, axis=1)

    print(x[0])

    model = Sequential()
    model.add(keras.layers.Dense(11, input_shape=(1, 11), activation='relu'))
    # model.add(keras.layers.Dense(110, activation='relu'))
    model.add(Dense(1, activation="linear"))

    model.compile(loss='mse', optimizer='adam')

    # model.fit(x, y, epochs=32, batch_size=1)
    # import math
    for txi, tyi, xi, yi in zip(x, y, x_, y_):
        # xhat = model.predict(xi.reshape(1, 1, 11))
        # print(model.predict(xhat), yi, xhat-xi)
        model.train_on_batch(txi.reshape(1, 1, 11),tyi)
        # model.train_on_batch(xi.reshape(1, 1, 11),yi)
    #
    # scores = model.evaluate(x_, y_)
    # print("%s: %.2f%%" % (model.metrics_names, scores * 100))

    return model

def search_space():
    return {
        'daytrader-config-app_CONMGR1_AGED_TIMEOUT',
        'daytrader-config-app_CONMGR1_MAX_IDLE_TIMEOUT',
        'daytrader-config-app_CONMGR1_MAX_POOL_SIZE',
        'daytrader-config-app_CONMGR1_MIN_POOL_SIZE',
        'daytrader-config-app_CONMGR1_REAP_TIME',
        'daytrader-config-app_CONMGR1_TIMEOUT',
        'daytrader-config-app_HTTP_MAX_KEEP_ALIVE_REQUESTS',
        'daytrader-config-app_HTTP_PERSIST_TIMEOUT',
        'daytrader-config-app_MAX_THREADS',
        'daytrader-service_cpu',
        'daytrader-service_memory'
    }

from pprint import pprint
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
if __name__ == '__main__':
    # df = load_configs('./resources/logging-trxrhel-202011261015.csv')

    df, tdf = load_configs('./resources/logging-trxrhel-202011241500.csv')
    # df = df.sample(frac=1).reset_index(drop=True)
    model = learning(df, tdf)


    df, tdf = load_configs('./resources/logging-trxrhel-202011221630.csv')
    # df = df.sample(frac=1).reset_index(drop=True)
    # keys, X, n = transform(df['configs'])
    # pprint(X)
    # Y = df['values']
    #
    # X = np.expand_dims(X, axis=1)
    # Y = np.expand_dims(Y, axis=1)
    #
    # print(model.evaluate(X, Y))

    columns = list(df.columns)
    columns.remove('values')

    x = tdf[columns].values
    y = tdf['values'].to_numpy()
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    scores = model.evaluate(x, y)
    print("%s: %.2f%%" % (model.metrics_names, scores * 100))

    counter = 1
    acc1 = 0
    acc2 = 0
    # curr = curr + (value - curr) / self.counter[best_idx]
    c1 = 0
    c2 = 0
    import math
    prediction = []
    measured = []
    avg = []
    for xi, yi in zip(x, y):
        yhat = model.predict(xi.reshape(1, 1, 11))
        prediction.append(yhat[0][0][0])
        measured.append(yi[0])
        acc1 += (yi[0] - acc1) / counter
        acc2 += (yhat[0][0][0] - acc2) / counter
        avg.append(acc1)

        # if predicted is lt measured avg
        # if math.fabs(yhat[0][0][0] - yi[0]) <= 0.05:
        if math.fabs(yhat[0][0][0] - acc1) >= -0.02:
            c1 += 1
            print(f'[{(yhat[0][0][0] - acc1) >= -0.02} yhat:{yhat[0][0][0]:0.02f} y:{yi[0]:0.02f}  diff:{yhat[0][0][0] - acc1:0.02f} avg:{acc1:0.2f} ')
            model.train_on_batch(xi.reshape(1, 1, 11), yi)
        else:
            if np.random.random() <= 0.3:
                model.train_on_batch(xi.reshape(1, 1, 11), yi)
            c2 += 1
            print(f'[{(yhat[0][0][0] - acc1) >= -0.02}] yhat:{yhat[0][0][0]:0.02f} y:{yi[0]:0.02f} diff:{yhat[0][0][0] - acc1:0.02f} avg:{acc1:0.2f} ')
        #
        # print(f'update:{math.fabs(yhat[0][0][0] - yi[0]) <= 0.05} yhat: {yhat[0][0][0]:0.2f} y: {yi[0]:0.2f}')
        counter += 1


    scores = model.evaluate(x, y)
    print("%.2f%% %s: %.2f%%" % ((c1/counter)*100, model.metrics_names, scores * 100))

    plt.plot(prediction, label='prediction', linewidth=0.7)
    plt.plot(measured, label='measured', linewidth=0.7)
    plt.plot(avg, label='avg', linewidth=0.7)
    plt.legend(['prediction', 'measured', 'avg'])
    plt.show()
