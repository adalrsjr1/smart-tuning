import os
import time
import json

from pymongo import MongoClient

import app_config
from seqkmeans import Container
import configsampler as cs
import workloadhandler as wh

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
    print('sampling workload')
    workload = wh.workload_and_metric(app_config.NAMESPACE, app_config.WAITING_TIME, app_config.MOCK)
    print('setting new configuration to the workload')
    workload.configuration = configuration
    workload.start = start
    workload.end = start + app_config.WAITING_TIME
    print('classifying the workload')
    workload.classification, hits = wh.classify(workload)

    return hits, workload

client = MongoClient(app_config.MONGO_ADDR, app_config.MONGO_PORT)
def save(workload:Container)->str:
    db = client[app_config.MONGO_DB]
    collection = db.tunning_collection
    return collection.insert_one(workload.serialize())

def best_tuning(id)->Container:
    pass

if __name__ == "__main__":
    while True:

        configuration = update_config()
        start = int(time.time())
        print(f'waiting {app_config.WAITING_TIME}s for a new workload')
        time.sleep(app_config.WAITING_TIME)

        hits, workload = classify_workload(configuration)

        save(workload)

        # sync
        # cluster_id, #hits
