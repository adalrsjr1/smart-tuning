import json
import os
import time

import pymongo

import config
import configsampler as cs
import sampler as wh
from seqkmeans import Container, KmeansContext

def update_config(last_metric)->dict:
    if config.MOCK:
        return {}

    search_space = cs.load_search_space(config.SEARCHSPACE_PATH)
    if last_metric:
        search_space.update_model(last_metric)

    config_map = cs.ConfigMap()

    print('sampling new configuration')
    configuration = search_space.sampling()
    print('new config >>> ', configuration)

    print('patching new config')
    config_map.patch(config.CONFIGMAP_NAME, config.NAMESPACE, configuration)
    # config_map.patch_jvm(config.CONFIGMAP_NAME, config.NAMESPACE, configuration)
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

def save_prod_metric(timestamp, prod_metric, metric):
    db = config.client[config.MONGO_DB]
    collection = db.prod_metric_collection
    return collection.insert_one({'time': timestamp, 'prod_metric':prod_metric, 'tuning_metric':metric})

def best_tuning(classification=None)->Container:
    db = config.client[config.MONGO_DB]
    collection = db.tuning_collection
    print('finding best tuning')
    best_item = next(collection.find({'classification':classification}).sort("metric", pymongo.DESCENDING).limit(1))

    return best_item['classification'], best_item['configuration'], best_item['metric']

def main():
    last_config, last_type, last_prod_metric = None, None, 0
    last_train_matric = None
    configMapHandler = cs.ConfigMap()

    start = time.time()
    print(f' *** waiting {config.WAITING_TIME}s for application warm up *** ')
    while time.time() - start < config.WAITING_TIME:
        print('.', end='')
        time.sleep(10)
    print()

    print(f'*** starting SmartTuning loop ***')
    first_loop = True
    iteration_counter = 1

    classificationCtx = KmeansContext(config.K)

    while True:
        start = time.time()
        configuration = update_config(last_train_matric)

        print(f' *** waiting {config.WAITING_TIME}s for a new workload *** ')
        time.sleep(config.WAITING_TIME)

        print(' *** sampling workloads *** ')
        print('\tsampling training workload')
        workload = wh.workload(config.POD_REGEX, int(config.WAITING_TIME * config.SAMPLE_SIZE)).result()
        # workload = wh.workload_and_metric(config.POD_REGEX, int(config.WAITING_TIME * config.SAMPLE_SIZE), config.MOCK)

        print('classifying workload ', workload.label)
        workload.classification, workload.hits = classificationCtx.cluster(workload)
        print(f'workload {workload.label} classified as {workload.classification.id} -- {workload.hits}th hit')


        workload.configuration = configuration

        last_train_matric = workload.metric

        print('\tsampling production workload')
        workload_prod = wh.workload_and_metric(config.POD_PROD_REGEX, int(config.WAITING_TIME * config.SAMPLE_SIZE), config.MOCK)
        workload_prod.classification = last_type
        workload_prod.configuration = last_config

        print(f'\t\t[T] throughput: {workload.metric}')
        print(f'\t\t[P] throughput: {workload_prod.metric}')

        if first_loop:
            first_loop = False
            print(f'smarttuning loop took {time.time() - start}s')
            continue

        print(' *** saving data ***')
        save(workload)
        save_prod_metric(workload.start, workload_prod.metric, workload.metric)
        save_workload(workload_prod, workload)


        print(' *** sampling the best tuning *** ')
        best_type, best_config, best_metric = best_tuning(workload.classification)
        print('\tbest type: ', best_type, '\n\tlast type: ', last_type)
        print('\tbest conf: ', best_config, '\n\tlast config: ', last_config)
        print('\tbest metric: ', best_metric, '\n\tlast metric: ', last_prod_metric)

        if best_config and '_id' in best_config:
            del (best_config['_id'])

        if last_config and '_id' in last_config:
            del (last_config['_id'])

        print('\n *** deciding about update the application *** \n')
        print(f'\tis the best config stable? {workload.hits > config.NUMBER_ITERATIONS}')
        if workload.hits >= config.NUMBER_ITERATIONS:
            print(f'\tis best metric > current prod metric? {best_metric > workload_prod.metric * (1+config.METRIC_THRESHOLD)}')
            if best_metric > workload_prod.metric * (1+config.METRIC_THRESHOLD):
                print(f'\tis last config != best config? ', last_config != best_config)
                if last_config != best_config:
                    print(f'setting config: {best_config}')
                    configMapHandler.patch(config.CONFIGMAP_PROD_NAME, config.NAMESPACE_PROD, best_config)
                    # configMapHandler.patch_jvm(config.CONFIGMAP_PROD_NAME, config.NAMESPACE_PROD, best_config)

                    last_config = best_config
                    save_config_applied(best_config or {})

        last_type = workload.classification
        last_prod_metric = workload_prod.metric

        print(f' *** smarttuning loop [{iteration_counter}] took {time.time() - start}s *** \n\n')
        iteration_counter += 1


if __name__ == '__main__':
    main()