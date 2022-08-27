import json
import math
import pandas as pd


def load(filenames: list[str], replica_type: str = 'training') -> pd.DataFrame:


    table = pd.DataFrame()
    for filename in filenames:
        with open(filename) as file:
            for row in file.readlines():
                data = json.loads(row)
                if is_same_workload(data):
                    df = pd.json_normalize(data[replica_type]['metric'] |
                                            {'score': data[replica_type]['curr_config']['score'],
                                            'config': data[replica_type]['curr_config']['name'],
                                            'workload': data['curr_workload']['name'],
                                            'status': data['status']},
                                            sep='_')

                    table = pd.concat([table, df], ignore_index=True)

    # column_names = ['daytrader-config-app_MAX_THREADS',
    #                 'daytrader-config-app_CONMGR1_MAX_POOL_SIZE',
    #                 'daytrader-config-app_CONMGR1_MIN_POOL_SIZE',
    #                 'daytrader-config-app_CONMGR1_TIMEOUT',
    #                 'daytrader-config-app_CONMGR1_AGED_TIMEOUT',
    #                 'daytrader-config-app_CONMGR1_MAX_IDLE_TIMEOUT',
    #                 'daytrader-config-app_CONMGR1_REAP_TIME',
    #                 'daytrader-config-app_CONMGR4_MAX_POOL_SIZE',
    #                 'daytrader-config-app_CONMGR4_MIN_POOL_SIZE',
    #                 'daytrader-config-app_CONMGR4_TIMEOUT',
    #                 'daytrader-config-app_CONMGR4_AGED_TIMEOUT',
    #                 'daytrader-config-app_CONMGR4_MAX_IDLE_TIMEOUT',
    #                 'daytrader-config-app_CONMGR4_REAP_TIME',
    #                 'daytrader-config-app_HTTP_MAX_KEEP_ALIVE_REQUESTS',
    #                 'daytrader-config-app_HTTP_PERSIST_TIMEOUT']
    # table[column_names] = table[column_names].astype(int)
    return table.convert_dtypes()


def is_same_workload(data: dict) -> bool:
    return data['ctx_workload']['name'] == data['curr_workload']['name']


def is_training(data: dict) -> bool:
    return 'TrainingIteration' == data['status']


def count_config_failures(df: pd.DataFrame, workload: str) -> int:
    return len(df.loc[(df['workload'] == workload) & (df['throughput'] <= 10) & (df['status'] == 'TrainingIteration')])


def first_update(df: pd.DataFrame, workload: str) -> int:
    df = df.loc[(df['workload'] == workload)].reset_index(drop=True)
    try:
        return df['config'].drop_duplicates(keep='first').index[1]
    except IndexError:
        return 0
    # return df['config'].ne(df['config'].iloc[0]).idxmax()


def total_updates(df: pd.DataFrame, workload: str) -> int:
    df = df.loc[(df['workload'] == workload) & (df['status'] == 'TrainingIteration')].reset_index(drop=True)
    return df['config'].nunique()-1


def total_reinforcements(df: pd.DataFrame, workload: str) -> int:
    df = df.loc[(df['workload'] == workload) & (df['status'] == 'ReinforcementIteration')].reset_index(drop=True)
    return df['config'].nunique()-1


def total_probations(df: pd.DataFrame, workload: str) -> int:
    df = df.loc[(df['workload'] == workload) & (df['status'] == 'ProbationIteration')].reset_index(drop=True)
    return df['config'].nunique()-1


def print_txt(filenames: list[str]):
    print(f'workload \t #failures \t first update \t total updates')
    for i, filename in enumerate(filenames):
        training = load([filename], replica_type='training')
        production = load([filename], replica_type='production')
        print(f'{i}-JSP', '\t\t', count_config_failures(training, 'workload_jsp'), '\t\t', first_update(production, 'workload_jsp'), '\t\t', total_updates(production, 'workload_jsp'))
        print(f'{i}-JSF', '\t\t', count_config_failures(training, 'workload_jsf'), '\t\t', first_update(production, 'workload_jsf'), '\t\t', total_updates(production, 'workload_jsf'))



if '__main__' == __name__:
    filenames = [
        './resources/trace-daytrader-2021-09-22T02 42 28.json',                 # 0 jsp-jsf original (no-contention)
        './resources/trace-daytrader-jspjsf-2022-08-13T15_51_49.json',          # 1 jsp-jsf default
        './resources/trace-daytrader-jspjsf-random-2022-08-15T13_43_41.json',   # 2 jsp-jsf random
        './resources/trace-daytrader-jspjsfd-2022-08-14T11_50_52.json',         # 3 jsp-jsf no dependency {"training.metric.throughput": {$lt: 10}}
        './resources/trace-daytrader-jspjsfv-2022-08-14T11_51_00.json',         # 4 jsp-jsf no vertical scaling
        './resources/trace-daytrader-jspjsfv-2022-08-17T12_24_16.json',         # 5 jsp-jsf no memory
        './resources/trace-daytrader-jspjsf-2022-08-20T16_05_52.json',          # 6 jsp-jsp no thread
        './resources/trace-daytrader-jspjsfv-2022-08-19T00_12_23.json',         # 7 jsp-jsf no memory and no thread
        './resources/trace-daytrader-jspjsfw-2022-08-13T15_52_05.json',         # 8 jsp-jsf weight 10 to resp time
        './resources/trace-daytrader-jspjsfw-2022-08-15T13_39_16.json',         # 9 jsp-jsf weight 1000 to resp time
    ]

    print_txt(filenames)

