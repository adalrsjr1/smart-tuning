import json
import re
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import pandas as pd


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
                    # df = pd.json_normalize(data['training']['curr_config']['data'] |
                    #                        {'score': data['training']['curr_config']['score'],
                    #                        'workload': data['curr_workload']['name']},
                    #                        sep='_')
                    df = pd.json_normalize({'score': data['training']['curr_config']['score']} |
                                           data['training']['metric'] |
                                           data['training']['curr_config']['data'],
                                           sep='_')

                    table = pd.concat([table, df], ignore_index=True)

    return table.fillna(0)


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def correlation(df: pd.DataFrame, parameters: list[str]) -> pd.DataFrame:
    df = df[parameters].astype(float)
    df.columns = clean_feature_names(df.columns)
    # for parameter in parameters:
        # df[parameter] = normalize_data(df[parameter])

    print(df.corr(method='pearson'))
    # pd.plotting.scatter_matrix(df, alpha=0.2)
    # plt.show()


def clean_feature_names(features: list[str]) -> list[str]:
    return [re.compile("cat__x[0-9]+_|-(app_|jvm|system_|service_)*").split(name)[-1] for name in features]


if '__main__' == __name__:
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 600
    # pd.options.display.max_colwidth = 10

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

    df = load(filenames)
    print(df.columns)
    # columns = clean_feature_names(df.columns)
    # df.columns = columns
    # correlation(df, set(df.columns).difference(['cpu', 'cpu_limit', 'memory_limit', 'penalization', 
    #                                             '_gc', 'daytrader-config-jvm_gc', '_container_support', 
    #                                             'Xtune:virtualized']))
    correlation(df, ['score', 'memory', 'throughput', 'process_time', 'daytrader-service_memory',
                     'daytrader-config-app_MAX_THREADS', 'daytrader-config-app_CONMGR1_MAX_POOL_SIZE',
                     'daytrader-config-app_CONMGR1_MIN_POOL_SIZE', 'daytrader-config-app_CONMGR4_MAX_POOL_SIZE',
                     'daytrader-config-app_CONMGR4_MAX_POOL_SIZE', 'daytrader-config-app_CONMGR1_TIMEOUT'])

    # for filename in filenames:
    #     df = load([filename])
    #     print(df.columns)
    #     correlation(df, ['memory', 'throughput', 'process_time', 'objective'])
    #     break