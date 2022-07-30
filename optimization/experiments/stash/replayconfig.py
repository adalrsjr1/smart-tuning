import pandas as pd
import hashlib
import kubernetes
import os, sys, time

import sampler

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

    df= pd.DataFrame({'names':configs_name,  'configs':configs, 'values':df['production_metric.objective'].values * -1,
                      'tnames':tconfigs_name, 'tconfigs':tconfigs, 'tvalues':df['training_metric.objective'].values * -1})
    return df


def sort_configs_by_frequency(df:pd.DataFrame) -> pd.DataFrame:
    df = df.assign(freq=df.apply(lambda x: df.names.value_counts() \
                            .to_dict()[x.names], axis=1)) \
        .sort_values(by=['freq', 'names','values'], ascending=[False, True, False]).loc[:,]

    df.reset_index(inplace=True)
    df.rename(columns={'index':'sorted_index0'}, inplace=True)

    df = df.sort_values(by=['freq', 'names', 'sorted_index0'], ascending=[False, True, True]).loc[:,].reset_index()
    return df.rename(columns={'index':'sorted_index1'})

def replay(n:int, df:pd.DataFrame, name:str, objective:str, waiting_time:int, sample_size:int):
    new_values = []
    for index, row in df.iterrows():
        if index < n:
            patch_config(row['configs'])
            time.sleep(waiting_time)
            value = sample(name, objective, waiting_time, sample_size)
            new_values.append(value)
        else:
            new_values.append(float('NaN'))

    return df.assign(new_values=new_values, axis=1)


def patch_config(name:str, namespace:str, config:dict):
    for manifest, values in config:
        if manifest == 'daytrader-config-app':
            patch_deployment(name, namespace)
        else:
            patch_configmap(name, namespace)

def patch_configmap(name:str, namespace:str, config:dict):
    # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#patch_namespaced_config_map
    body = {
        "kind": "ConfigMap",
        "apiVersion": "v1",
        "metadata": {"labels": {"date": str(int(time.time()))}},
        "data": config
    }

    return config.coreApi().patch_namespaced_config_map(name, namespace, body, pretty='true')

def patch_deployment(name:str, namespace:str, config:dict):
    appsApi = kubernetes.client.AppsV1Api()
    api_instance = config.appsApi()

    deployment = appsApi.read_namespaced_deployment(name=name, namespace=namespace)
    containers = deployment.spec.template.spec.containers

    total_cpu = config.get('cpu', None)
    total_memory = config.get('memory', None)

    container: kubernetes.client.V1Container
    for container in containers:
        resources: kubernetes.client.V1ResourceRequirements = container.resources

        if not resources:
            resources = kubernetes.client.V1ResourceRequirements(limits={}, requests={})

        limits = resources.limits
        if not limits:
            limits = {}

        if total_cpu:
            limits.update({
                'cpu': str(total_cpu / len(containers))
            })

        if total_memory:
            limits.update({
                'memory': str(int(total_memory / len(containers))) + 'Mi'
            })

        container.resources.limits = limits

    body = {
        "kind": "Deployment",
        "apiVersion": "v1",
        "metadata": {"labels": {"date": str(int(time.time()))}},
        "spec": {"template": {"spec": {"containers": containers}}}
    }

    return api_instance.patch_namespaced_deployment(name, namespace, body, pretty='true')

def sample(name:str, objective:str, waiting_time:int, sample_size:float) -> sampler.Metric:
    prom = sampler.PrometheusSampler(name, waiting_time * sample_size)
    return prom.metric(to_eval=objective).objective()

from pprint import pprint
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    # df = load_configs('./resources/logging-trxrhel-202011221630.csv')
    df = load_configs('./resources/logging-trxrhel-202011261015.csv')
    sorted_df = sort_configs_by_frequency(df)
    sorted_df.to_csv(r'./resources/logging-trxrhel-202011261015_sorted.csv')
    print(sorted_df)

    # objective = '-(throughput / ((((memory / (2**20)) * 0.013375) + (cpu * 0.0535) ) / 2))'
    # waiting_time = 1200
    # sample_size = 0.3334
    # df = replay(sorted_df, objective, waiting_time, sample_size)

    # print(df.to_json())

