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
    start = int(time.time())
    print('sampling workload')
    workload = wh.workload_and_metric(config.POD_REGEX, config.WAITING_TIME, config.MOCK)
    print('set new configuration to the workload')
    workload.configuration = configuration
    workload.start = start
    workload.end = start + config.WAITING_TIME
    workload.classification, hits = wh.classify(workload)

    return hits, workload

def save(workload:Container)->str:
    db = config.client[config.MONGO_DB]
    collection = db.tuning_collection
    return collection.insert_one(workload.serialize())


def best_tuning(type=None)->Container:
    db = config.client[config.MONGO_DB]
    collection = db.tuning_collection
    print('finding best tuning')
    best_item = next(collection.find({'classification':type}).sort("metric", pymongo.DESCENDING).limit(1))

    return best_item['classification'], best_item['configuration'], best_item['metric']

def suitable_config(new_id, new_config, new_metric, old_id, old_config, old_metric, hits, convergence=10, metric_threshold=0.2):
    print('checking if tuning can be applied')
    if hits < convergence:
        print('\tconfiguration does not converged yet')
        return old_config

    if old_metric is None or new_metric > old_metric * (1 + metric_threshold):
        print('\trelevant improvement')
        if new_id != old_id:
            print('\tnew config is different configuration from actual')
            return new_config
    print('\tmaintaining current configuration')
    return old_config

def sync(tuning_config, endpoints):
    for endpoint in endpoints.values():
        print(f'syncing config at {endpoint}')
        requests.post(f'http://{endpoint}:{config.SYNC_PORT}/reload', data=tuning_config)

def main():
    register.start()
    last_config, last_type = None, None
    while True:
        if config.MOCK:
            configuration = {}
        else:
            configuration = update_config()

        print(f'waiting {config.WAITING_TIME}s for a new workload')
        time.sleep(config.WAITING_TIME)

        hits, workload = classify_workload(configuration)
        last_metric = workload.metric
        workload.hits = hits
        # save current workload
        save(workload)
        # fetch best configuration

        best_type, best_config, best_metric = best_tuning(workload.classification)
        config_to_apply = suitable_config(best_type, best_config, best_metric, last_type, last_config, last_metric, hits,
                                          convergence=config.NUMBER_ITERATIONS, metric_threshold=config.METRIC_THRESHOLD)


        print(f'>>> hits:{hits}, best_type:{best_type}, best_config:{config_to_apply}')
        if config_to_apply != last_config:
            print('do sync')
            sync(config_to_apply, register.list())
            last_config = config_to_apply
            last_type = best_type

        else:
            print('do nothing: last config is the best config')

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