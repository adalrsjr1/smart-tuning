import json
import os
import time

import pymongo

import config
import configsampler as cs
import workloadhandler as wh
from seqkmeans import Container


def update_config()->dict:
    if config.MOCK:
        return {}

    envvar = {}
    files = [f for f in os.listdir(config.CONFIG_PATH) if os.path.isfile(os.path.join(config.CONFIG_PATH, f))]

    for f in files:
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
    config_map.patch(config.CONFIGMAP_NAME, config.NAMESPACE, configuration)
    return configuration

def save(workload:Container)->str:
    db = config.client[config.MONGO_DB]
    collection = db.tuning_collection
    return collection.insert_one(workload.serialize())

def save_workload(workload_prod, workload_training):
    db = config.client[config.MONGO_DB]
    collection = db.workloads_collection
    return collection.insert_one({'prod_workload': workload_prod.serialize(), 'training_workload': workload_training.serialize()})

def save_config_applied(config_applied):
    db = config.client[config.MONGO_DB]
    collection = db.configs_applied_collection
    return collection.insert_one(config_applied)

def save_prod_metric(prod_metric, metric):
    db = config.client[config.MONGO_DB]
    collection = db.prod_metric_collection
    return collection.insert_one({'prod_metric':prod_metric, 'tuning_metric':metric})

def best_tuning(classification=None)->Container:
    db = config.client[config.MONGO_DB]
    collection = db.tuning_collection
    print('finding best tuning')
    best_item = next(collection.find({'classification':classification}).sort("metric", pymongo.DESCENDING).limit(1))

    return best_item['classification'], best_item['configuration'], best_item['metric']

def main():
    last_config, last_type, last_prod_metric = None, None, 0
    configMapHandler = cs.ConfigMap()

    start = time.time()
    print(f'waiting {config.WAITING_TIME}s for the application warming up')
    while time.time() - start < config.WAITING_TIME:
        print('.', end='')
        time.sleep(10)
    print()

    first_loop = True
    while True:
        start = time.time()
        configuration = update_config()

        print(f'waiting {config.WAITING_TIME}s for a new workload')
        time.sleep(config.WAITING_TIME)

        print('sampling workloads')
        print('sampling training workload')
        workload = wh.workload_and_metric(config.POD_REGEX, config.WAITING_TIME, config.MOCK)
        workload.classification, workload.hits = wh.classify(workload)
        workload.configuration = configuration

        print('sampling production workload')
        workload_prod = wh.workload_and_metric(config.POD_PROD_REGEX, config.WAITING_TIME, config.MOCK)
        workload_prod.classification = last_type
        workload_prod.configuration = last_config

        print(f'\t[T] throughput: {workload.metric}')
        print(f'\t[P] throughput: {workload_prod.metric}')


        if first_loop:
            first_loop = False
            print(f'smarttuning loop took {time.time() - start}s')
            continue

        print('saving data')
        save(workload)
        save_prod_metric(workload_prod.metric, workload.metric)
        save_workload(workload_prod, workload)

        print('sampling the best tuning')
        best_type, best_config, best_metric = best_tuning(workload.classification)
        print('\tbest type: ', best_type, '\n\tlast type: ', last_type)
        print('\tbest conf: ', best_config, '\n\tlast config: ', last_config)
        print('\tbest metric: ', best_metric, '\n\tlast metric: ', last_prod_metric)

        print('\ndeciding about update the application\n')
        print(f'\tis the best config stable? {workload.hits > config.NUMBER_ITERATIONS}')
        if workload.hits > config.NUMBER_ITERATIONS:
            print(f'\tis best metric > current prod metric? {best_metric > workload_prod.metric * (1+config.METRIC_THRESHOLD)}')
            if best_metric > workload_prod.metric * (1+config.METRIC_THRESHOLD):
                print(f'\tis last config != best config? ', last_config != best_config)
                if last_config != best_config:
                    print(f'setting config: {best_config}')
                    configMapHandler.patch(config.CONFIGMAP_PROD_NAME, config.NAMESPACE_PROD, best_config)


                    if best_config and '_id' in best_config:
                        del (best_config['_id'])

                    if last_config and '_id' in last_config:
                        del (last_config['_id'])

                    last_config = best_config
                    save_config_applied(best_config or {})

        last_type = workload.classification
        last_prod_metric = workload_prod.metric

        print(f'smarttuning loop took {time.time() - start}s')


if __name__ == '__main__':
    main()