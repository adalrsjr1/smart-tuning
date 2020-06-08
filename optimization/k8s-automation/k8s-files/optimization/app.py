import time

import pymongo
import heapq
import config
import configsampler as cs
from concurrent.futures import Future
import sampler as wh
from seqkmeans import Container, KmeansContext, Metric, Cluster

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

    return collection.insert_one({'time': timestamp, 'prod_metric':prod_metric.serialize(), 'tuning_metric':metric.serialize()})

def best_tuning(classification:Cluster, tuning_candidates:list)->Container:
    for container in tuning_candidates:
        if container.classification == classification:
            return container.classification, container.configuration, container.metric

    return None

def metrics_result(cpu:Future, memory:Future, throughput:Future, latency:Future):
    metric = Metric()
    cpu = cpu.result().replace(float('NaN'), 0)
    metric.cpu = cpu[0] if not cpu.empty else 0

    memory = memory.result().replace(float('NaN'), 0)
    metric.memory = memory[0] if not memory.empty else 0

    throughput = throughput.result().replace(float('NaN'), 0)
    metric.throughput = throughput[0] if not throughput.empty else 0

    latency = latency.result().replace(float('NaN'), 0)
    metric.latency = latency[0] if not latency.empty else 0

    return metric

def main():
    tuning_candidates = []
    last_config, last_type, last_prod_metric = None, None, 0
    last_train_matric = None
    configMapHandler = cs.ConfigMap()

    start = time.time()
    waiting_time = config.WAITING_TIME * config.SAMPLE_SIZE
    print(f' *** waiting {waiting_time}s for application warm up *** ')
    while time.time() - start < waiting_time:
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
        print('Training workload:')
        workload = wh.workload(pod_regex=config.POD_REGEX,
                               interval=int(config.WAITING_TIME * config.SAMPLE_SIZE))

        print('\nTraining metrics')
        latency = wh.latency(pod_regex=config.POD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                             quantile=config.QUANTILE)

        throughput = wh.throughput(pod_regex=config.POD_REGEX,
                                   interval=int(config.WAITING_TIME * config.SAMPLE_SIZE), quantile=config.QUANTILE)

        memory = wh.memory(pod_regex=config.POD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                           quantile=config.QUANTILE)

        cpu = wh.cpu(pod_regex=config.POD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                     quantile=config.QUANTILE)

        workload = Container(str(int(time.time())), workload.result())
        workload.metric = metrics_result(cpu, memory, throughput, latency)

        print('classifying workload ', workload.label)
        workload.classification, workload.hits = classificationCtx.cluster(workload)
        print(f'\tworkload {workload.label} classified as {workload.classification.id} -- {workload.hits}th hit')

        workload.configuration = configuration

        last_train_matric = workload.metric

        print('Production workload:')
        workload_prod = wh.workload(pod_regex=config.POD_PROD_REGEX,
                                    interval=int(config.WAITING_TIME * config.SAMPLE_SIZE))

        print('\nTraining metrics:')
        latency_prod = wh.latency(pod_regex=config.POD_PROD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                             quantile=config.QUANTILE)

        throughput_prod = wh.throughput(pod_regex=config.POD_PROD_REGEX,
                                   interval=int(config.WAITING_TIME * config.SAMPLE_SIZE), quantile=config.QUANTILE)

        memory_prod = wh.memory(pod_regex=config.POD_PROD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                           quantile=config.QUANTILE)

        cpu_prod = wh.cpu(pod_regex=config.POD_PROD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                     quantile=config.QUANTILE)

        workload_prod = Container(str(int(time.time())), workload_prod.result())
        workload_prod.metric = metrics_result(cpu_prod, memory_prod, throughput_prod, latency_prod)

        workload_prod.classification = last_type
        workload_prod.configuration = last_config

        print(f'\t\t[T] metrics: {workload.metric}')
        print(f'\t\t[P] metrics: {workload_prod.metric}')

        # if first_loop:
        #     first_loop = False
        #     print(f'smarttuning loop tooks {time.time() - start}s')
        #     continue

        print(' *** saving data ***')
        heapq.heappush(tuning_candidates, workload)
        save(workload)
        save_prod_metric(workload.start, workload_prod.metric, workload.metric)
        save_workload(workload_prod, workload)


        print(' *** sampling the best tuning *** ')
        best_type, best_config, best_metric = best_tuning(workload.classification, tuning_candidates)
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
            is_min = best_metric < workload_prod.metric * (1+config.METRIC_THRESHOLD)
            print(f'\tminimization: is best metric < current prod metric? {is_min}')
            if is_min:
                print(f'\tis last config != best config? ', last_config != best_config)
                if last_config != best_config:
                    print(f'setting config: {best_config}')
                    configMapHandler.patch(config.CONFIGMAP_PROD_NAME, config.NAMESPACE_PROD, best_config)
                    # configMapHandler.patch_jvm(config.CONFIGMAP_PROD_NAME, config.NAMESPACE_PROD, best_config)

                    last_config = best_config
                    save_config_applied(best_config or {})

        last_type = workload.classification
        last_prod_metric = workload_prod.metric

        print(f' *** smarttuning loop [{iteration_counter}] tooks {time.time() - start}s *** \n\n')
        iteration_counter += 1


if __name__ == '__main__':
    main()