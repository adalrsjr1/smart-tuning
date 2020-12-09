import datetime
import logging
import numbers
import time
from concurrent.futures import Future

import numpy as np

import config
import sampler
from bayesian import BayesianDTO
from controllers import injector, searchspace
from controllers.k8seventloop import EventLoop
from controllers.searchspace import SearchSpaceContext
from seqkmeans import Container, KmeansContext, Metric, Cluster

logger = logging.getLogger(config.APP_LOGGER)
logger.setLevel('DEBUG')


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
        samplers[training] = sampler.PrometheusSampler(training, config.WAITING_TIME * config.SAMPLE_SIZE)


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
    production:str
    for production, training in microservices.items():
        if not production in contexts:
            logger.debug(f'creating contexts for {microservices} in {contexts}')
            contexts[production] = config.executor().submit(create_context, production, training)

    # TODO: need further improvements
    to_remove = []
    for microservice, future in contexts.items():
        if future.done():
            logger.debug(f'gc: marking to remove {microservice} ctx')
            to_remove.append(microservice)

    for microservice in to_remove:
        logger.debug(f'gc contexts: wiping {microservice} ctx')
        del contexts[microservice]

    if 0 == len(microservices) < len(contexts):
        future:Future
        for future in contexts.values():
            future.cancel()

from controllers.planner import Planner
from models.instance import Instance
from models.configuration import Configuration, EmptyConfiguration


def create_context(production_microservice, training_microservice):
    logger.info(f'creating context for ({production_microservice},{training_microservice})')
    production_name = production_microservice
    training_name = training_microservice

    production_sanitized = production_microservice.replace('.', '\\.')
    training_sanitized = training_microservice.replace('.', '\\.')
    logger.info(f'pod_names --  prod:{production_sanitized} train:{training_sanitized}')


    while production_name and training_name:
        try:
            search_space_ctx: SearchSpaceContext = sample_config(production_name)
            if not search_space_ctx:
                return

            if search_space_ctx:
                production = Instance(name=production_sanitized, namespace=config.NAMESPACE, is_production=True,sample_interval_in_secs=config.WAITING_TIME * config.SAMPLE_SIZE, ctx=search_space_ctx)
                training = Instance(name=training_sanitized, namespace=config.NAMESPACE, is_production=False,sample_interval_in_secs=config.WAITING_TIME * config.SAMPLE_SIZE, ctx=search_space_ctx)
                p = Planner(production=production, training=training, ctx=search_space_ctx, k=config.ITERATIONS_BEFORE_REINFORCE, ratio=config.REINFORCEMENT_RATIO)

                configuration: Configuration
                for i in range(config.NUMBER_ITERATIONS):
                    configuration, last_iteration = next(p)
                    logger.info(f'[{i}, last:{last_iteration}] {configuration}')

                # if last_iteration or isinstance(configuration, EmptyConfiguration):
                logger.warning(f'stoping bayesian core for {production.name}')

                searchspace.stop_tuning(search_space_ctx)
                del training

        except:
            logger.exception('error during tuning iteration')
        finally:
            exit(0)

## depecrated
def create_context2(production_microservice, training_microservice):
    logger.info(f'creating context for ({production_microservice},{training_microservice})')
    production_name = production_microservice
    training_name = training_microservice

    production_sanitized = production_microservice.replace('.', '\\.')
    training_sanitized = training_microservice.replace('.', '\\.')
    logger.info(f'pod_names --  prod:{production_sanitized} train:{training_sanitized}')
    sampler_production = sampler.PrometheusSampler(production_sanitized, config.WAITING_TIME * config.SAMPLE_SIZE)
    sampler_training = sampler.PrometheusSampler(training_sanitized, config.WAITING_TIME * config.SAMPLE_SIZE)

    last_config = {}
    config_to_apply = {}
    first_config = {}
    last_class = None
    first_iteration = True
    while production_name and training_name:


        try:
            search_space_ctx: SearchSpaceContext = sample_config(production_name)



            if not search_space_ctx:
                logger.debug(f'no searchspace configuration available for microservice {production_name}')
                time.sleep(1)
                continue

            if len(last_config) == 0:
                last_config = search_space_ctx.get_current_config()

            if first_iteration:
                first_iteration = False
                first_config = last_config
                production_workload = sampler_production.workload()
                production_metric = sampler_production.metric()
                save(
                    timestamp=datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
                    last_config=last_config,
                    trials=get_trials(search_space_ctx),
                    production_metric=production_metric.serialize(),
                    config_to_eval={},
                    training_metric={},
                    production_workload=sampler.series_to_dict(
                        production_workload.result(config.SAMPLING_METRICS_TIMEOUT)),
                    # extract to dict
                    training_workload={},
                    # extract to dict
                    best_loss=production_metric.objective(),
                    best_config=config_to_apply,
                    update_production=config_to_apply,
                    tuned=False
                )

            config_to_apply = update_training_config(training_microservice, search_space_ctx)

            if ('is_best_config', True) in config_to_apply.items():
                logger.info(f'applying best config {config_to_apply}')
                best_config, best_loss = best_loss_so_far(search_space_ctx)
                logger.info('deleting training pod ')
                delete_deployment(training_microservice)
                update_production(production_microservice, best_config, search_space_ctx)
                logger.info('waiting for definitive measurements')
                time.sleep(config.WAITING_TIME)

                production_workload = sampler_production.workload()
                production_metric = sampler_production.metric()

                save(
                    timestamp=datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
                    last_config=last_config,
                    trials=get_trials(search_space_ctx),
                    production_metric=production_metric.serialize(),
                    config_to_eval={},
                    training_metric={},
                    production_workload=sampler.series_to_dict(
                        production_workload.result(config.SAMPLING_METRICS_TIMEOUT)),
                    # extract to dict
                    training_workload={},
                    # extract to dict
                    best_loss=production_metric.objective(),
                    best_config=config_to_apply,
                    update_production=config_to_apply,
                    tuned=config_to_apply!=last_config
                )
                logger.info(f'stoping tuning for microservice:{production_microservice}')
                search_space_ctx.delete_bayesian_searchspace()
                return


            fail_fast, production_metric, training_metric = wait(True, config.WAITING_TIME, production_sanitized, training_sanitized)

            if production_metric != Metric.zero() and training_metric != Metric.zero():
                production_metric = sampler_production.metric()
                training_metric = sampler_training.metric()

            production_workload = sampler_production.workload()
            training_workload = sampler_training.workload()

            production_result = production_workload.result(config.SAMPLING_METRICS_TIMEOUT)
            production_class, production_hits = classify_workload(production_metric, production_result)

            training_result = training_workload.result(config.SAMPLING_METRICS_TIMEOUT)
            training_class, training_hits = classify_workload(training_metric, training_result)

            if last_config != first_config:
                update_best_loss(search_space_ctx, production_metric.objective())

            # if fail_fast:
            #     logger.info('fail fast')
            #     loss = update_loss(training_class, Metric.zero(), search_space_ctx)
            # else:
            loss = update_loss(training_class, training_metric, search_space_ctx)
                # loss = update_loss(production_class, production_metric, search_space_ctx)

            time.sleep(2)  # to avoid race condition when iterating over Trials
            best_config, best_loss = best_loss_so_far(search_space_ctx)


            evaluation = training_metric.objective() <= production_metric.objective() * (1 - config.METRIC_THRESHOLD)
            logger.info(f'[t metric] {training_metric}')
            logger.info(f'[p metric] {production_metric}')
            logger.info(f'[objective] training:{training_metric.objective()} production:{production_metric.objective()} best:{best_loss}')

            logger.info(
                f'training:{training_metric.objective()} <= production:{production_metric.objective()} '
                f'| loss:{loss} <= best_loss:{best_loss}')
            tuned = False
            updated_config = last_config

            last_class = production_class
            if best_loss <= production_metric.objective() * (1 - config.METRIC_THRESHOLD):
                logger.info(f'[curr] best:{best_loss} <= production:{production_metric.objective()}')
                logger.info(f'[last] {last_config}')
                logger.info(f'[best] {best_config}')
                if last_config != best_config:
                    updated_config = best_config
                    update_production(production_microservice, updated_config, search_space_ctx)
                    tuned = 'best_config'
                else:
                    tuned = 'last_config'
            else:
                logger.info(f'[curr] best:{best_loss} > production:{production_metric.objective()}')
                tuned = 'false'



            # if evaluation and best_loss >= loss:
            #     # if last_config != config_to_apply or last_class != production_class:
            #     logger.info(
            #         f'[curr] training:{training_metric.objective()} <= production:{production_metric.objective()} '
            #         f'| best_loss:{best_loss} >= loss:{loss}'
            #         f'== {evaluation and (best_loss >= loss)}')
            #     last_class = production_class
            #     update_production(production_microservice, config_to_apply, search_space_ctx)
            #     updated_config = config_to_apply
            #     tuned = 'training_config'
            # else:
            #     old_best_config, old_best_loss = best_config, best_loss
            #     # update_best_loss(search_space_ctx, production_metric.objective())
            #     # best_config, best_loss = best_loss_so_far(search_space_ctx)
            #
            #     if best_loss < production_metric.objective():
            #         logger.info(
            #             f'[best] training:{training_metric.objective()} <= production:{production_metric.objective()} '
            #             f'| best_loss:{best_loss} --> prod:{production_metric.objective()}'
            #             f'== {evaluation and (best_loss < loss)}')
            #
            #         last_class = production_class
            #
            #         update_production(production_microservice, best_config, search_space_ctx)
            #         updated_config = best_config
            #
            #         if updated_config != last_config:
            #             tuned = 'previous_config'
            #         else:
            #             tuned = False
            #     else:
            #         tuned = False

            save(
                timestamp=datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
                last_config=last_config,
                trials=get_trials(search_space_ctx),
                production_metric=production_metric.serialize(),
                config_to_eval=config_to_apply,
                training_metric=training_metric.serialize(),
                production_workload=sampler.series_to_dict(production_workload.result(config.SAMPLING_METRICS_TIMEOUT)),
                # extract to dict
                training_workload=sampler.series_to_dict(training_workload.result(config.SAMPLING_METRICS_TIMEOUT)),
                # extract to dict
                best_loss=best_loss,
                best_config=best_config,
                update_production=updated_config,
                tuned=tuned
            )
            last_config = updated_config
        except:
            logger.exception(f'error when handling microservice({production_microservice},{training_microservice})')




def sample_config(microservice) -> SearchSpaceContext:
    logger.debug(f'lookup {microservice} in {searchspace.search_spaces.keys()}')
    return searchspace.search_spaces.get(f'{microservice}', None)


def update_training_config(name, new_config_ctx: SearchSpaceContext):
    config_to_apply = new_config_ctx.get_from_engine()
    logger.info(f'updating training microservice {name} with config {config_to_apply}')

    # manifests = new_config_ctx.manifests
    manifests = new_config_ctx.model.manifests

    do_patch(manifests, config_to_apply)
    return config_to_apply


def do_patch(manifests, configuration, production=False):
    for key, value in configuration.items():
        for manifest in manifests:
            logger.info(f'checking to patch {key} into {manifest._name}')
            if key == manifest._name:
                logger.info(f'patching new config={value} at {manifest._name}')
                manifest.patch(value, production=production)
                time.sleep(1)

def delete_deployment(name:str):
    config.appsApi().delete_namespaced_deployment(name, config.NAMESPACE)


classificationCtx = KmeansContext(config.K)


def classify_workload(metric, workload) -> (Cluster, int):
    label = str(time.time_ns())
    logger.info(f'classifying workload {label}')
    container = Container(label=label, content=workload, metric=metric)
    return classificationCtx.cluster(container)


def update_loss(classification: Cluster, metric_value: Metric, search_space_ctx: SearchSpaceContext):
    logger.info(f'updating loss at BayesianEngine: {search_space_ctx.engine.id()}')
    dto = BayesianDTO(metric=metric_value, workload_classification= classification.id if classification else None)
    search_space_ctx.put_into_engine(dto)
    if metric_value:
        return metric_value.objective()
    return Metric.zero().objective()

def best_loss_so_far(search_space_ctx: SearchSpaceContext):
    logger.info(f'getting best loss at BayesianEngine: {search_space_ctx.engine.id()}')
    return search_space_ctx.get_best_so_far()

def update_best_loss(search_space_ctx: SearchSpaceContext, new_loss:float):
    logger.info(f'updating best loss at trials to {new_loss}')
    search_space_ctx.update_best_loss(new_loss)

def update_production(name, config, search_space_ctx: SearchSpaceContext):
    logger.info(f'updating production microservice {name} with config {config}')
    manifests = search_space_ctx.model.manifests
    do_patch(manifests, config, production=True)

def get_trials(search_space_ctx: SearchSpaceContext):
    if search_space_ctx:
        return search_space_ctx.get_trials_as_documents()
    return {}


def wait(failfast=True, sleeptime=config.WAITING_TIME, production_pod_name='', training_pod_name=''):
    if not failfast:
        time.sleep(sleeptime)
        return False, Metric.zero(), Metric.zero()

    sampler_production = sampler.PrometheusSampler(production_pod_name, sleeptime * config.SAMPLE_SIZE)
    sampler_training = sampler.PrometheusSampler(training_pod_name, sleeptime * config.SAMPLE_SIZE)
    now = time.time()
    counter = 1
    production_metric, training_metric = None, None
    while (time.time() - now) < sleeptime:
        logger.info(f'[{counter}] waiting {sleeptime * config.SAMPLE_SIZE}s before sampling metrics -- time:{(time.time() - now):.2f} < sleeptime:{sleeptime:.2f}')
        time.sleep(sleeptime * config.SAMPLE_SIZE)
        production_metric = sampler_production.metric()
        training_metric = sampler_training.metric()
        logger.info(f'\t\_ prod:{production_metric.objective():.2f}  train:{training_metric.objective():.2f}')

        # objective is always negative
        # training < 50% production
        if counter > 1:
            if training_metric.throughput() <= config.THROUGHPUT_THRESHOLD or production_metric.objective()/2 < training_metric.objective():
                # training fail fast
                logger.info(f'[T] fail fast -- prod:{production_metric.objective()} < train:{training_metric.objective()}')
                return True, production_metric, training_metric
            elif production_metric.throughput() <= config.THROUGHPUT_THRESHOLD or training_metric.objective()/2 < production_metric.objective():
                # production fail fast
                logger.info(f'[P] fail fast -- prod:{production_metric.objective()} >= train:{training_metric.objective()}')
                return False, production_metric, training_metric
            else:
                logger.info(f'waiting more {sleeptime * config.SAMPLE_SIZE}s -- prod:{production_metric.objective()} >= train:{training_metric.objective()}')
        counter += 1
    # time.sleep(sleeptime)

    return False, production_metric, training_metric

def save(**kwargs):
    logger.info(f'logging data to mongo')
    config.mongo().admin
    if not config.ping(config.MONGO_ADDR, config.MONGO_PORT):
        logger.warning(f'cannot save logging -- mongo unable at {config.MONGO_ADDR}:{config.MONGO_PORT}')
        return None
    db = config.mongo()[config.MONGO_DB]
    collection = db.logging

    try:
        collection.insert_one(sanitize_dict_to_save(kwargs))
    except:
        logger.exception('error when saving data')

def sanitize_dict_to_save(data) -> dict:
    result = {}

    for item in data.items():
        key = sanitize_token(item[0])
        value = sanitize_token(item[1])
        result.update({key: value})

    return result

def sanitize_list_to_save(data):
    result = []

    for item in data:
        result.append(sanitize_token(item))

    if isinstance(data, list):
        return result
    if isinstance(data, set):
        return set(result)
    if isinstance(data, tuple):
        return tuple(data)

def sanitize_token(token) -> str:
    if isinstance(token, str):
        return token.replace( '\\', '\\\\').replace('$', '\\$').replace( '.', '\\_')
    if isinstance(token, dict):
        return sanitize_dict_to_save(token)
    if isinstance(token, list) or isinstance(token, set) or isinstance(token, tuple):
        return sanitize_list_to_save(token)
    if isinstance(token, numbers.Number) or isinstance(token, bool):
        return token
    return sanitize_dict_to_save(token.__dict__)

##
# all services ports should be named
# all deployments and services should be annotated with 'injection.smarttuning.ibm.com'
# kubectl apply -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
# kubectl delete -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
def main():
    init()
    while True:
        # logger.info('into main loop')
        try:
            # duplicate microservices

            # create optimization contexts
            create_contexts(get_microservices())
            time.sleep(1)
        except:
            logger.exception(f'error on main loop')
            break

# https://github.com/kubernetes-client/python/blob/cef5e9bd10a6d5ca4d9c83da46ccfe2114cdaaf8/examples/notebooks/intro_notebook.ipynb
# repactor injector using this approach

# TODO: Change training and production pod without restart them
# TODO: Stop iteration when throughput goes to 0 or below a given threshold
if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('main loop error')
    finally:
        config.shutdown()
