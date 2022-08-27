from email.utils import collapse_rfc2231_value
import json
import re
from pprint import pprint
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


def load(filenames: list[str]) -> pd.DataFrame:
    def is_same_workload(data: dict) -> bool:
        return data['ctx_workload']['name'] == data['curr_workload']['name']

    def is_training(data: dict) -> bool:
        return 'TrainingIteration' == data['status']

    table = pd.DataFrame()
    for filename in filenames:
        with open(filename) as file:
            for row in file.readlines():
                data = json.loads(row)
                if is_training(data) and is_same_workload(data):
                    df = pd.json_normalize(data['training']['curr_config']['data'] |
                                           {'score': data['training']['curr_config']['score'],
                                           'workload': data['curr_workload']['name']},
                                           sep='_')

                    table = pd.concat([table, df], ignore_index=True)

    column_names = ['daytrader-config-app_MAX_THREADS',
                    'daytrader-config-app_CONMGR1_MAX_POOL_SIZE',
                    'daytrader-config-app_CONMGR1_MIN_POOL_SIZE',
                    'daytrader-config-app_CONMGR1_TIMEOUT',
                    'daytrader-config-app_CONMGR1_AGED_TIMEOUT',
                    'daytrader-config-app_CONMGR1_MAX_IDLE_TIMEOUT',
                    'daytrader-config-app_CONMGR1_REAP_TIME',
                    'daytrader-config-app_CONMGR4_MAX_POOL_SIZE',
                    'daytrader-config-app_CONMGR4_MIN_POOL_SIZE',
                    'daytrader-config-app_CONMGR4_TIMEOUT',
                    'daytrader-config-app_CONMGR4_AGED_TIMEOUT',
                    'daytrader-config-app_CONMGR4_MAX_IDLE_TIMEOUT',
                    'daytrader-config-app_CONMGR4_REAP_TIME',
                    'daytrader-config-app_HTTP_MAX_KEEP_ALIVE_REQUESTS',
                    'daytrader-config-app_HTTP_PERSIST_TIMEOUT']
    table[column_names] = table[column_names].astype(int)
    return table.convert_dtypes()


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def importance(df: pd.DataFrame, prune: list[str], bootstraping: int):
    # df is table of all parameters, one parameter per column, one iteration per row
    # dropping columns
    df = df.drop(prune, axis=1)
    df = df.fillna(0.0)
    df = resample(df, n_samples=len(df) * bootstraping, random_state=123)

    # https://www.datacamp.com/community/tutorials/xgboost-in-python
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, 0]

    # one-hot encode the categorical features
    cat_attribs = [item for item in ['daytrader-config-jvm_gc', 'workload'] if item not in prune]
    full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)],
                                      remainder='passthrough')
    encoder = full_pipeline.fit(X)
    X = encoder.transform(X)
    X = pd.DataFrame(X, columns=encoder.get_feature_names_out()).astype(int)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
    #                          max_depth=5, alpha=10, n_estimators=100)
    model = xgb.XGBRegressor()

    model.fit(X, y)
    features_names = encoder.get_feature_names_out()
    features_scores = model.feature_importances_

    scores = model.get_booster().get_fscore()
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[0])}   
    return scores


def plotter(ax: Axes, scores: pd.DataFrame):
    for idx, column in enumerate(scores.columns):
        normalized_score = normalize_data(scores[column]) * 200
        ax.scatter([idx]*len(scores), scores.index, s=normalized_score, alpha=0.5)

    ax.set_xticks(np.arange(len(scores.columns)))
    ax.set_yticks([x for x in np.arange(len(scores))])
    ax.set_yticklabels(clean_feature_names(scores.index), fontsize='small')
    ax.grid(True)

    return ax


def clean_feature_names(features: list[str]) -> list[str]:
    return [re.compile("cat__x[0-9]+_|-(app_|jvm|system_|service_)*").split(name)[-1] for name in features]


if '__main__' == __name__:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    filenames = [
        './resources/trace-daytrader-2021-09-22T02 42 28.json',                 # 0 jsp-jsf original (no contention)
        './resources/trace-daytrader-jspjsf-2022-08-13T15_51_49.json',          # 1 jsp-jsf default
        './resources/trace-daytrader-jspjsf-random-2022-08-15T13_43_41.json',   # 2 jsp-jsf random
        './resources/trace-daytrader-jspjsfd-2022-08-14T11_50_52.json',         # 3 jsp-jsf no dependency {"training.metric.throughput": {$lt: 10}}
        './resources/trace-daytrader-jspjsfv-2022-08-14T11_51_00.json',         # 4 jsp-jsf no vertical scaling
        './resources/trace-daytrader-jspjsfv-2022-08-17T12_24_16.json',         # 5 jsp-jsf no memory
        './resources/trace-daytrader-jspjsf-2022-08-20T16_05_52.json',          # 6 jsp-jsf no thread
        './resources/trace-daytrader-jspjsfv-2022-08-19T00_12_23.json',         # 7 jsp-jsf no memory and no thread
        './resources/trace-daytrader-jspjsfw-2022-08-13T15_52_05.json',         # 8 jsp-jsf weight 10 to resp time
        './resources/trace-daytrader-jspjsfw-2022-08-15T13_39_16.json',         # 9 jsp-jsf weight 1000 to resp time
    ]

    ax = None
    fig = None
    fig, ax = plt.subplots()

    concatened_df = pd.DataFrame()
    # one experiment per column
    for idx, filename in enumerate(filenames):
        df = load([filename])

        scores = importance(df, prune=['workload',
                                       # 'daytrader-service_memory',
                                       'daytrader-service_cpu',
                                       # 'daytrader-config-app_MAX_THREADS',
                                       # 'daytrader-config-app_CONMGR1_MAX_POOL_SIZE',
                                       # 'daytrader-config-app_CONMGR1_MIN_POOL_SIZE',
                                       'daytrader-config-jvm_gc',
                                       'daytrader-config-jvm_-Xtune:virtualized'],
                            bootstraping=1000)
        concatened_df = concatened_df.append(scores, ignore_index=True)

    # bag all experiments together
    df = load(filenames)
    scores = importance(df, prune=['workload',
                                   # 'daytrader-service_memory',
                                   'daytrader-service_cpu',
                                   # 'daytrader-config-app_MAX_THREADS',
                                   # 'daytrader-config-app_CONMGR1_MAX_POOL_SIZE',
                                   # 'daytrader-config-app_CONMGR1_MIN_POOL_SIZE',
                                   'daytrader-config-jvm_gc',
                                   'daytrader-config-jvm_-Xtune:virtualized'],
                        bootstraping=200)
    concatened_df = concatened_df.append(scores, ignore_index=True)

    # split all bagged experiments per workload 'jsp'
    df_jsp = df.loc[df['workload'] == 'workload_jsp']
    scores = importance(df, prune=['workload',
                                   # 'daytrader-service_memory',
                                   'daytrader-service_cpu',
                                   # 'daytrader-config-app_MAX_THREADS',
                                   # 'daytrader-config-app_CONMGR1_MAX_POOL_SIZE',
                                   # 'daytrader-config-app_CONMGR1_MIN_POOL_SIZE',
                                   'daytrader-config-jvm_gc',
                                   'daytrader-config-jvm_-Xtune:virtualized'],
                        bootstraping=200)
    concatened_df = concatened_df.append(scores, ignore_index=True)

    # split all bagged experiments per workload 'jsf'
    df_jsp = df.loc[df['workload'] == 'workload_jsf']
    scores = importance(df, prune=['workload',
                                   # 'daytrader-service_memory',
                                   'daytrader-service_cpu',
                                   # 'daytrader-config-app_MAX_THREADS',
                                   # 'daytrader-config-app_CONMGR1_MAX_POOL_SIZE',
                                   # 'daytrader-config-app_CONMGR1_MIN_POOL_SIZE',
                                   'daytrader-config-jvm_gc',
                                   'daytrader-config-jvm_-Xtune:virtualized'],
                        bootstraping=200)
    concatened_df = concatened_df.append(scores, ignore_index=True)

    plotter(ax, concatened_df.reset_index(drop=True).T.sort_index())

    print(concatened_df.T.to_string())

    ax.set_title('Parameter importance')
    ax.set_xlabel('Experiment #')
    fig.tight_layout()
    plt.show()
