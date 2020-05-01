import os
import time
import json

import pymongo

import app_config
from seqkmeans import Container
import configsampler as cs
import workloadhandler as wh
import register
import requests

def update_config()->dict:
    print('loading config file')
    envvar = {}
    files = [f for f in os.listdir(app_config.CONFIG_PATH) if os.path.isfile(os.path.join(app_config.CONFIG_PATH, f))]

    for f in files:
        envvar[f] = os.environ.get(f, '')

    search_space = cs.SearchSpace({})
    config_map = cs.ConfigMap()

    print('loading search space')
    with open(app_config.SEARCHSPACE_PATH) as json_file:
        data = json.load(json_file)
        for item in data:
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
    config_map.patch(app_config.CONFIGMAP_NAME, app_config.NAMESPACE, configuration)

    return configuration

def classify_workload(configuration:dict)->(int, Container):
    """
    :param configuration:
    :return: how many times this type was hit and the type (as a workload)
    """
    start = int(time.time())
    print('sampling workload')
    workload = wh.workload_and_metric(app_config.NAMESPACE, app_config.WAITING_TIME, app_config.MOCK)
    print('setting new configuration to the workload')
    workload.configuration = configuration
    workload.start = start
    workload.end = start + app_config.WAITING_TIME
    print('classifying the workload')
    workload.classification, hits = wh.classify(workload)

    return hits, workload

def save(workload:Container)->str:
    db = app_config.client[app_config.MONGO_DB]
    collection = db.tuning_collection
    return collection.insert_one(workload.serialize())

from bson.son import SON
def best_tuning(type=None)->Container:
    db = app_config.client[app_config.MONGO_DB]
    collection = db.tuning_collection

    best_item = next(collection.find({'classification':type}).sort("metric", pymongo.DESCENDING).limit(1))

    return best_item['classification'], best_item['configuration'], best_item['metric']

def suitable_config(new_id, new_config, new_metric, old_id, old_config, old_metric, hits, convergence=10, metric_threshold=0.2):

    if hits < convergence:
        return old_config

    if old_metric is None or new_metric > old_metric * (1 + metric_threshold):
        if new_id != old_id:
            return new_config
    return old_config

def sync(config, endpoints):
    for endpoint in endpoints.values():
        requests.post(f'{endpoint}:{app_config.SYNC_PORT}/reload', data=config)

if __name__ == "__main__":
    register.start()
    last_config, last_metric, last_type = None, None, None
    while True:
        if app_config.MOCK:
            configuration = {}
        else:
            configuration = update_config()


        print(f'waiting {app_config.WAITING_TIME}s for a new workload')
        time.sleep(app_config.WAITING_TIME)

        hits, workload = classify_workload(configuration)

        # save current workload
        save(workload)
        # fetch best configuration

        best_type, best_config, best_metric = best_tuning(workload.classification)
        config_to_apply = suitable_config(best_type, best_config, best_metric, last_type, last_config, last_metric, hits,
                                          convergence=app_config.NUMBER_ITERATIONS, metric_threshold=app_config.METRIC_THRESHOLD)

        print('>>> ', hits, best_type, best_config)
        sync(config_to_apply)

        if hits >= app_config.NUMBER_ITERATIONS:
            print('do sync')
            sync(best_config, register.list())
