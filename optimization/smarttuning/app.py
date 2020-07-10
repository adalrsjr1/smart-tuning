import logging
import time

import numpy as np

import config
import sampler
import kubernetes
from concurrent.futures import Future
from bayesian import BayesianDTO
from controllers import injector, searchspace
from controllers.k8seventloop import EventLoop
from controllers.searchspace import SearchSpaceContext
from seqkmeans import Container, KmeansContext, Metric, Cluster

logger = logging.getLogger(config.APP_LOGGER)
logger.setLevel(logging.DEBUG)


def warmup():
    start = time.time()
    waiting_time = config.WAITING_TIME * config.SAMPLE_SIZE
    logger.info(f'waiting {waiting_time}s for application warm up')

    time_counter = 0
    while time.time() - start < waiting_time:
        logger.debug(f'counter: {time_counter}')
        time.sleep(10)
        time_counter += 10

    return time.time() - start


microservices: dict = {}


def microservices_listener():
    logger.info('starting microservices listener')
    global microservices
    while True:
        new_microservices = injector.duplicated_dep.keys()
        for microservice in new_microservices:
            if not microservice in microservices:
                logger.info(f'initializing microservice {microservice}')
                microservices[microservice] = {
                    'kmeans': KmeansContext(config.K),
                    'status': False
                }

        time.sleep(1)


def initialize_samplers(samplers, production, training):
    if not production in samplers:
        samplers[production] = sampler.PrometheusSampler(production, config.WAITING_TIME * config.SAMPLE_SIZE)

    if not training in samplers:
        samplers[production] = sampler.PrometheusSampler(training, config.WAITING_TIME * config.SAMPLE_SIZE)


def microservice_loop(microservice):
    waiting = np.random.randint(1, 10)
    while True:
        print(microservice, waiting)
        time.sleep(waiting)


def init():
    event_loop = EventLoop(config.executor())
    # initializing controllers
    searchspace.init(event_loop)
    injector.init(event_loop)


def get_microservices():
    return injector.duplicated_dep


contexts = {}


def create_contexts(microservices):
    logger.info(f'creating contexts for {microservices} in {contexts}')
    production:str
    for production, training in microservices.items():

        if not production in contexts:
            contexts[production] = config.executor().submit(create_context, production, training)

    # TODO: need further improvements
    to_remove = []
    for microservice, future in contexts.items():
        if future.done():
            logger.info(f'gc: marking to remove {microservice} ctx')
            to_remove.append(microservice)

    for microservice in to_remove:
        logger.info(f'gc contexts: wiping {microservice} ctx')
        del contexts[microservice]

    if 0 == len(microservices) < len(contexts):
        future:Future
        for future in contexts.values():
            future.cancel()


def create_context(production_microservice, training_microservice):
    logger.info(f'creating context for ({production_microservice},{training_microservice})')
    production_name = production_microservice
    training_name = training_microservice

    production_sanitized = production_microservice.replace('.', '\\.')
    training_sanitized = training_microservice.replace('.', '\\.')

    sampler_production = sampler.PrometheusSampler(production_sanitized, config.WAITING_TIME * config.SAMPLE_SIZE)
    sampler_training = sampler.PrometheusSampler(training_sanitized, config.WAITING_TIME * config.SAMPLE_SIZE)

    last_config = None
    last_class = None
    while production_name and training_name:
        try:
            search_space_ctx: SearchSpaceContext = sample_config(production_name)
            if not search_space_ctx:
                logger.info(f'no searchspace configuration available for microservice {production_name}')
                time.sleep(1)
                continue
            config_to_apply = update_training_config(training_microservice, search_space_ctx)
            # wait(config.WAITING_TIME)

            production_metric = sampler_production.metric()
            production_workload = sampler_production.workload()
            training_metric = sampler_training.metric()
            training_workload = sampler_training.workload()

            production_class, production_hits = classify_workload(production_metric, production_workload.result(
                config.SAMPLING_METRICS_TIMEOUT))
            training_class, training_hits = classify_workload(training_metric,
                                                              training_workload.result(config.SAMPLING_METRICS_TIMEOUT))

            loss = update_loss(training_class, training_metric, search_space_ctx)

            time.sleep(2)  # to avoid race condition when iterating over Trials
            best_config, best_loss = best_loss_so_far(search_space_ctx)
            if loss <= best_loss * (1 - config.METRIC_THRESHOLD):
                if last_config != best_config or last_class != production_class:
                    update_production(production_microservice, best_config, search_space_ctx)
                    last_config = best_config
                    last_class = production_class

            save(
                timestamp=time.time_ns(),
                config=config_to_apply,
                production_metric=production_metric.serialize(),
                training_metric=training_metric.serialize(),
                production_workload=sampler.series_to_dict(production_workload.result(config.SAMPLING_METRICS_TIMEOUT)),
                # extract to dict
                training_workload=sampler.series_to_dict(training_workload.result(config.SAMPLING_METRICS_TIMEOUT)),
                # extract to dict
                best_loss=best_loss,
                best_config=best_config,
                update_production=best_config
            )

        except:
            logger.exception(f'error when handling microservice({production_microservice},{training_microservice})')


def sample_config(microservice) -> SearchSpaceContext:
    return searchspace.search_spaces.get(microservice, None)


def update_training_config(name, new_config_ctx: SearchSpaceContext):
    config_to_apply = new_config_ctx.get_from_engine()
    logger.info(f'updating training microservice {name} with config {config_to_apply}')

    manifests = new_config_ctx.manifests
    do_patch(manifests, config_to_apply)
    return config_to_apply


def do_patch(manifests, configuration, production=False):
    for key, value in configuration.items():
        for manifest in manifests:
            if key == manifest.name:
                logger.debug(f'patching new config at {manifest.name}')
                manifest.patch(value, production=production)


classificationCtx = KmeansContext(config.K)


def classify_workload(metric, workload) -> (Cluster, int):
    label = str(time.time_ns())
    logger.info(f'classifying workload {label}')
    container = Container(label=label, content=workload, metric=metric)
    return classificationCtx.cluster(container)


def update_loss(classification: Cluster, metric_value: Metric, search_space_ctx: SearchSpaceContext):
    logger.info(f'updating loss at BayesianEngine: {search_space_ctx.engine.id()}')
    dto = BayesianDTO(metric=metric_value, classification=classification.id)
    search_space_ctx.put_into_engine(dto)
    return metric_value.objective()


def best_loss_so_far(search_space_ctx: SearchSpaceContext):
    logger.info(f'getting best loss at BayesianEngine: {search_space_ctx.engine.id()}')
    return search_space_ctx.get_best_so_far()


def update_production(name, config, search_space_ctx: SearchSpaceContext):
    logger.info(f'updating production microservice {name} with config {config}')
    manifests = search_space_ctx.manifests
    do_patch(manifests, config, production=True)


def save(**kwargs):
    logger.info(f'logging data to mongo')
    config.mongo().admin
    if not config.ping(config.MONGO_ADDR, config.MONGO_PORT):
        logger.warning(f'cannot save logging -- mongo unable at {config.MONGO_ADDR}:{config.MONGO_PORT}')
        return None
    db = config.mongo()[config.MONGO_DB]
    collection = db.logging

    return collection.insert_one(kwargs)

##
# all services ports should be named
# all deployments and services should be annotated with 'injection.smarttuning.ibm.com'
# kubectl apply -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
# kubectl delete -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
def main():
    init()
    while True:
        try:
            create_contexts(get_microservices())
            time.sleep(1)
        except:
            logger.exception(f'error on main loop')
            break


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('main loop error')
    finally:
        config.shutdown()
