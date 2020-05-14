import json
import os
import sys
import time
import random

import pymongo
import requests

import config
import configsampler as cs
import register
import workloadhandler as wh
from seqkmeans import Container


def update_config()->dict:
    envvar = {}
    files = [f for f in os.listdir(config.CONFIG_PATH) if os.path.isfile(os.path.join(config.CONFIG_PATH, f))]

    print('\n params to observe')
    for f in files:
        print('\t', f)
        envvar[f] = os.environ.get(f, '')

    search_space = cs.SearchSpace({})
    config_map = cs.ConfigMap()

    print('\nloading search space')
    with open(config.SEARCHSPACE_PATH) as json_file:
        data = json.load(json_file)
        for item in data:
            print('\t', item)
            search_space.add_to_domain(
                key=item.get('key', None),
                lower=item.get('lower', None),
                upper=item.get('upper', None),
                options=item.get('options', None),
                type=item.get('type', None)
            )

    print('sampling new configuration')
    configuration, length = search_space.sampling('sampling_label')
    configuration = search_space.sample_values_to_str(configuration)
    print('new config >>> ', configuration)

    print('patching new config')
    patch_result = config_map.patch(config.CONFIGMAP_NAME, config.NAMESPACE, configuration)
    return configuration

def classify_workload(configuration:dict)->(int, Container):
    """
    :param configuration:
    :return: how many times this type was hit and the type (as a workload)
    """
    end = int(time.time())
    print('sampling workload')
    workload = wh.workload_and_metric(config.POD_REGEX, config.WAITING_TIME, config.MOCK)
    print('set new configuration to the workload')
    workload.configuration = configuration
    workload.start = end - config.WAITING_TIME
    workload.end = end
    workload.classification, hits = wh.classify(workload)

    return hits, workload

def save(workload:Container)->str:
    db = config.client[config.MONGO_DB]
    collection = db.tuning_collection
    return collection.insert_one(workload.serialize())

def save_config_applied(config_applied):
    db = config.client[config.MONGO_DB]
    collection = db.configs_applied_collection
    return collection.insert_one(config_applied)

def save_prod_metric(prod_metric, metric):
    db = config.client[config.MONGO_DB]
    collection = db.prod_metric_collection
    return collection.insert_one({'prod_metric':prod_metric, 'tuning_metric':metric})

def best_tuning(type=None)->Container:
    db = config.client[config.MONGO_DB]
    collection = db.tuning_collection
    print('finding best tuning')
    best_item = next(collection.find({'classification':type}).sort("metric", pymongo.DESCENDING).limit(1))

    return best_item['classification'], best_item['configuration'], best_item['metric']

def suitable_config(new_id, new_config, new_metric, old_id, old_config, old_metric, hits, convergence=10, metric_threshold=0.2):
    print('checking if tuning can be applied')
    print(f'hits:{hits} < convergence:{convergence} == ', hits < convergence)
    if hits < convergence:
        print('\tconfiguration does not converged yet')
        return old_config

    print(f'new_metric:{new_metric} > old_metric:{old_metric} * (1 + {metric_threshold})', old_metric is None or new_metric > old_metric * (1 + metric_threshold))
    if old_metric is None or new_metric > old_metric * (1 + metric_threshold):
        print('\trelevant improvement')
        print(f'new_id:{new_id} != old_id:{old_id}')
        if config.K > 1 and (new_id != old_id or old_config is None):
            print('\tnew config is different configuration from actual')
            return new_config
        elif config.K <= 1 or old_config is None:
            return new_config
    print('\tmaintaining current configuration')
    return old_config

def sync(tuning_config, endpoints):
    for endpoint in endpoints.values():
        print(f'syncing config:{tuning_config} at {endpoint}')
        try:
            if tuning_config:
                response = requests.post(f'http://{endpoint}:{config.SYNC_PORT}/reload', data=tuning_config)
                print(response.content, response.status_code)
            else:
                print('config to sync is None')
        except Exception as e:
            print(e)

def main():
    last_config, last_type, last_prod_metric = None, None, 0
    configMapHandler = cs.ConfigMap()
    while True:
        if config.MOCK:
            configuration = {}
        else:
            configuration = update_config()

        print(f'waiting {config.WAITING_TIME}s for a new workload')
        time.sleep(config.WAITING_TIME)

        hits, workload = classify_workload(configuration)
        # get metric from production pod
        production_metric = wh.throughput(config.POD_PROD_REGEX, 60)
        training_metric = wh.throughput(config.POD_REGEX, 60)
        workload.metric = training_metric
        save_prod_metric(production_metric, training_metric)
        workload.hits = hits
        # save current workload
        save(workload)
        # fetch best configuration

        best_type, best_config, best_metric = best_tuning(workload.classification)
        config_to_apply = suitable_config(best_type, best_config, best_metric, last_type, last_config, production_metric, hits,
                                          convergence=config.NUMBER_ITERATIONS, metric_threshold=config.METRIC_THRESHOLD)
        print('suitable config: ', config_to_apply)

        print(f'>>> hits:{hits}, best_metric:{best_metric}, best_type:{best_type}, best_config:{config_to_apply}')
        # think better about all possibilities here
        # constraint the update by config_to_apply != last_config is unreallistic
        # the prod application may be stucked in a poor config
        if production_metric < last_prod_metric or config_to_apply != last_config:
            print('update production pod')

            configMapHandler.patch(config.CONFIGMAP_PROD_NAME, config.NAMESPACE_PROD, config_to_apply)
            # sync(config_to_apply, register.list())

            last_config = config_to_apply
            last_type = best_type
            last_prod_metric = production_metric
            # workaround for consistence
            if config_to_apply and '_id' in config_to_apply:
                del(config_to_apply['_id'])
            save_config_applied(config_to_apply or {})
        else:
            print('do nothing: last config is the best config')

        if production_metric > last_prod_metric:
            last_prod_metric = production_metric

if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except Exception as e:
    #     print('Interrupted: ', e)
    #     config.shutdown()
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)