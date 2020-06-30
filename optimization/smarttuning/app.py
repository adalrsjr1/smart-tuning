import heapq
import time
from concurrent.futures import Future

import config
import logging
import sampler as wh
import pandas as pd
import numpy as np
from seqkmeans import Container, KmeansContext, Metric, Cluster
import bayesian
from controllers import smarttuninginjector, searchspacemodel

logger = logging.getLogger(config.APP_LOGGER)
logger.setLevel(logging.DEBUG)

def init_bayesian():
    if not bayesian.running:
        manifests, search_space = searchspacemodel.init(cdr_search_space_name=config.SEARCH_SPACE_NAME, namespace=config.NAMESPACE)

        bayesian.init(search_space)

        return manifests, search_space

    logger.warning('bayesian module is already intialized')
    return None, None

def update_config(last_metric, manifests) -> dict:

    if last_metric:
        bayesian.put(last_metric)

    # update manifests
    logger.debug('sampling new configuration')
    configuration = dict(bayesian.get().items())
    do_patch(manifests, configuration, production=False)

    return configuration, manifests


def do_patch(manifests, configuration, production=False):
    for key, value in configuration.items():
        for manifest in manifests:
            if key == manifest.name:
                logger.debug(f'patching new config at {manifest.name}')
                manifest.patch(value, production=production)


def save_iteration(workload: Container) -> str:
    db = config.client[config.MONGO_DB]
    collection = db.tuning_collection
    return collection.insert_one(workload.serialize())


def save_workload(workload_prod, workload_training):
    db = config.client[config.MONGO_DB]
    collection = db.workloads_collection
    return collection.insert_one(
        {'prod_workload': workload_prod.serialize(), 'training_workload': workload_training.serialize()})


def save_config_applied(config_applied):
    db = config.client[config.MONGO_DB]
    collection = db.configs_applied_collection
    return collection.insert_one(config_applied)


def save_metrics(timestamp, prod_metric, train_metric, def_metric):
    db = config.client[config.MONGO_DB]
    collection = db.metric_collection

    return collection.insert_one(
        {
            'time': timestamp,
            'prod_metric': prod_metric.serialize(),
            'train_metric': train_metric.serialize(),
            'def_metric': def_metric.serialize()
         }
    )

def save(workload_train, workload_prod, tuning_candidates, def_metrics):
    logger.debug('saving data')
    workload_train = add_candidate(workload_train, tuning_candidates)
    save_iteration(workload_train)
    save_metrics(workload_train.start, workload_prod.metric, workload_train.metric, metrics_result(*def_metrics))
    save_workload(workload_prod, workload_train)

def best_tuning(classification: Cluster, tuning_candidates: list) -> Container:
    for container in tuning_candidates:
        if container.classification == classification:
            return container

    logger.warning('there is not best tunning, returning None')
    return None

def remove_candidate(classification: Cluster, tuning_candidates: list):
    selected = None
    for container in tuning_candidates:
        if container.classification == classification:
            selected = container
            break

    if selected:
        logger.debug(f'removing {selected} from tuning candidates set')
        tuning_candidates.remove(selected)

def add_candidate(candidate: Container, tuning_candidates: list) -> Container:
    """ add candidate avoid repetions"""
    equals = [candidate]
    for container in tuning_candidates:
        if container.configuration == candidate.configuration and container.classification == candidate.classification:
            heapq.heappush(equals, container)

    best = heapq.heappop(equals)
    for container in equals:
        tuning_candidates.remove(container)

    heapq.heappush(tuning_candidates, best)
    return best

def metrics_result(cpu: Future, memory: Future, throughput: Future, latency: Future, timeout=config.SAMPLING_METRICS_TIMEOUT):
    metric = Metric()
    try:
        cpu = cpu.result(timeout=timeout).replace(float('NaN'), 0)
        metric.cpu = cpu[0] if not cpu.empty else 0
    except TimeoutError:
        logger.exception('Timeout when sampling cpu')
        metric.cpu = float('inf')

    try:
        memory = memory.result(timeout=timeout).replace(float('NaN'), 0)
        metric.memory = memory[0] if not memory.empty else 0
    except TimeoutError:
        logger.exception('Timeout when sampling memory')
        metric.memory = float('inf')

    try:
        throughput = throughput.result(timeout=timeout).replace(float('NaN'), 0)
        metric.throughput = throughput[0] if not throughput.empty else 0
    except TimeoutError:
        logger.exception('Timeout when sampling throuhgput')
        metric.throughput = 0

    try:
        latency = latency.result(timeout=timeout).replace(float('NaN'), 0)
        metric.latency = latency[0] if not latency.empty else 0
    except TimeoutError:
        logger.exception('Timeout when sampling latency')
        metric.throughput = float('inf')

    return metric

def default_metrics(podname:str) -> Metric:
    latency = wh.latency(pod_regex=podname,
                              interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                              quantile=config.QUANTILE)

    throughput = wh.throughput(pod_regex=podname,
                                    interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                                    quantile=config.QUANTILE)

    memory = wh.memory(pod_regex=podname, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                            quantile=config.QUANTILE)

    cpu = wh.cpu(pod_regex=podname, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                      quantile=config.QUANTILE)

    return cpu, memory, throughput, latency

def sample_train_workload(classificationCtx:KmeansContext, local_configuration:dict, timeout=config.SAMPLING_METRICS_TIMEOUT):
    logger.info('sampling training workload')
    workload = wh.workload(pod_regex=config.POD_REGEX,
                           interval=int(config.WAITING_TIME * config.SAMPLE_SIZE))

    logger.info('sampling training metrics')
    latency = wh.latency(pod_regex=config.POD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                         quantile=config.QUANTILE)

    throughput = wh.throughput(pod_regex=config.POD_REGEX,
                               interval=int(config.WAITING_TIME * config.SAMPLE_SIZE), quantile=config.QUANTILE)

    memory = wh.memory(pod_regex=config.POD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                       quantile=config.QUANTILE)

    cpu = wh.cpu(pod_regex=config.POD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                 quantile=config.QUANTILE)

    try:
        workload_train = Container(str(int(time.time())), workload.result(timeout=timeout))
    except TimeoutError as e:
        logger.exception('timeout when sampling training workload')
        workload_train = Container(str(int(time.time())), pd.Series(dtype=np.float))

    workload_train.start = int(time.time() - config.WAITING_TIME)
    workload_train.metric = metrics_result(cpu, memory, throughput, latency)

    logger.info(f'classifying workload {workload_train.label}')
    workload_train.classification, workload_train.hits = classificationCtx.cluster(workload_train)
    logger.debug(
        f'\tworkload {workload_train.label} classified as {workload_train.classification.id} -- {workload_train.hits}th hit')
    workload_train.configuration = local_configuration

    return workload_train

def sample_prod_workload(classification, last_config, timeout=config.SAMPLING_METRICS_TIMEOUT):
    logger.info('sampling production workload')
    workload_prod = wh.workload(pod_regex=config.POD_PROD_REGEX,
                                interval=int(config.WAITING_TIME * config.SAMPLE_SIZE))

    logger.info('sampling production metrics')
    latency_prod = wh.latency(pod_regex=config.POD_PROD_REGEX,
                              interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                              quantile=config.QUANTILE)

    throughput_prod = wh.throughput(pod_regex=config.POD_PROD_REGEX,
                                    interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                                    quantile=config.QUANTILE)

    memory_prod = wh.memory(pod_regex=config.POD_PROD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                            quantile=config.QUANTILE)

    cpu_prod = wh.cpu(pod_regex=config.POD_PROD_REGEX, interval=int(config.WAITING_TIME * config.SAMPLE_SIZE),
                      quantile=config.QUANTILE)

    try:
        workload_prod = Container(str(int(time.time())), workload_prod.result(timeout=timeout))
    except TimeoutError as e:
        logger.exception('timeout when sampling prod workload')
        workload_prod = Container(str(int(time.time())), pd.Series(dtype=np.float))
    workload_prod.metric = metrics_result(cpu_prod, memory_prod, throughput_prod, latency_prod)

    workload_prod.classification = classification
    workload_prod.configuration = last_config

    return workload_prod

def main():
    smarttuninginjector.init()

    tuning_candidates = []
    last_config, last_type, last_prod_metric = None, None, 0
    last_train_metric = None

    start = time.time()
    waiting_time = config.WAITING_TIME * config.SAMPLE_SIZE
    logger.info(f'waiting {waiting_time}s for application warm up')

    time_counter = 0
    while time.time() - start < waiting_time:
        logger.debug(f'counter: {time_counter}')
        time.sleep(10)
        time_counter += 10

    logger.info('starting SmartTuning loop')
    first_loop = True
    iteration_counter = 1

    classificationCtx = KmeansContext(config.K)
    manifests, _searchspace = init_bayesian()
    while True:
        start = time.time()
        local_configuration, manifests = update_config(last_train_metric, manifests)
        logger.debug(f'local config:{local_configuration}')

        logger.info(f'waiting {config.WAITING_TIME}s for a new workload')
        time.sleep(config.WAITING_TIME)

        logger.debug('sampling training workload')
        workload_train = sample_train_workload(classificationCtx, local_configuration)

        logger.debug('sampling default metrics')
        def_metrics = default_metrics('acmeair-tuningdefault-.*')

        last_train_metric = workload_train.metric

        logger.debug('sampling production workload')
        workload_prod = sample_prod_workload(workload_train.classification, last_config)

        logger.info(f'[T] metrics: {workload_train.metric}')
        logger.info(f'[P] metrics: {workload_prod.metric}')

        if first_loop:
            first_loop = False
            logger.info(f'smarttuning loop tooks {time.time() - start}s at its first loop')
            continue

        save(workload_train, workload_prod, tuning_candidates, def_metrics)

        logger.debug('sampling the best tuning so far')
        best_workload = best_tuning(workload_train.classification, tuning_candidates)
        best_type = best_workload.classification
        best_config = best_workload.configuration
        best_metric = best_workload.metric
        logger.info('best type: ', best_type)
        logger.info('last type: ', last_type)
        logger.info('best conf: ', best_config)
        logger.info('last config: ', last_config)
        logger.info('best metric: ', best_metric)
        logger.info('last metric: ', last_prod_metric)

        # to remove this on next housekeeping
        # if best_config and '_id' in best_config:
        #     del (best_config['_id'])
        #
        # if last_config and '_id' in last_config:
        #     del (last_config['_id'])

        logger.debug('deciding about update the application')
        logger.debug(f'is the best config stable? {is_type_stable(workload_train)}')
        if is_type_stable(workload_train):
            is_min = is_best_metric(best_workload, workload_prod)
            logger.debug(f'minimization: is best metric < current prod metric? {is_min}')
            if is_min:
                logger.debug(f'is last config != best config? ', last_config != best_config)
                if last_config != best_config:
                    logger.debug(f'setting best global config: {best_config}')
                    last_config = do_adapt(manifests, best_config)
                #
                # else:
                #     is_local_min = is_best_metric(workload_train, workload_prod)
                #     print(f'\tis training metric < prod metric? ', is_local_min)
                #     if is_local_min:
                #         print('removing old best tuning')
                #         remove_candidate(workload_train.classification, tuning_candidates)
                #         print(f'setting best local config: {local_configuration}')
                #
                #         last_config = do_adapt(manifests, local_configuration)

        last_type = workload_prod.classification
        last_prod_metric = workload_prod.metric

        logger.info(f'smarttuning loop [{iteration_counter}] tooks {time.time() - start}s')
        iteration_counter += 1

def is_type_stable(workload:Container, limit:int=config.NUMBER_ITERATIONS):
    return workload.hits >= limit

def is_best_metric(best_workload:Container, workload_prod:Container, threshould=config.METRIC_THRESHOLD):
    # minimization
    if best_workload is None:
        return False
    return best_workload.metric <= workload_prod.metric * (1 + threshould)

def do_adapt(manifests, best_config, production=True):
    do_patch(manifests, best_config, production=production)
    save_config_applied(best_config or {})
    return best_config

if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('main loop error')
    finally:
        config.shutdown()
