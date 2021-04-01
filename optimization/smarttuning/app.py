from __future__ import annotations

import logging
import time
from datetime import datetime

import config
from controllers import injector, searchspace, workloadctrl
from controllers.k8seventloop import EventLoop
from controllers.searchspace import SearchSpaceContext
from models.configuration import Configuration
from models.instance import Instance
from smarttuning2.planner.iteration import IterationDriver, Iteration, TunedIteration

logger = logging.getLogger(config.APP_LOGGER)
logger.setLevel('DEBUG')


def init():
    event_loop = EventLoop(config.executor())
    # initializing controllers
    # TODO: create an event bus to handle the event comming from these watchers
    searchspace.init(event_loop)
    injector.init(event_loop)
    workloadctrl.init(event_loop)


contexts = {}


def create_contexts(microservices):
    production: str
    for production, training in microservices.items():
        if production not in contexts:
            logger.debug(f'creating contexts for {microservices} in {contexts}')
            # create a context for every microservice annoted to be tuned
            #   injection.smarttuning.ibm.com: "true"
            # and with search space deployed accordingly
            contexts[production] = config.executor().submit(create_context, production, training)


def create_context(production_microservice, training_microservice):
    logger.info(f'creating context for ({production_microservice},{training_microservice})')
    production_name = production_microservice

    production_sanitized = production_microservice.replace('.', '\\.')
    training_sanitized = training_microservice.replace('.', '\\.')
    logger.info(f'pod_names --  prod:{production_sanitized} train:{training_sanitized}')

    stoped = False

    drivers = {}
    uid = datetime.utcnow().isoformat()
    while not stoped:
        try:
            # get event from search space deployment
            with searchspace.search_space_lock:
                logger.debug(f'trying to get event regarding {production_name} ss')
                event = searchspace.search_spaces.get(production_name)

            # busy waiting if search space wasn't deployed yet
            # TODO: get ride of this event handler
            # TODO: use an event bus to handle this
            if not event:
                logger.info(f'{production_name} waiting for search space')
                time.sleep(0.1)
                continue

            if 'DELETED' == event['type']:
                logger.warning(f'stoping smarrtuning tuning for app: {production_name}')
                stoped = True
                logger.warning('search space removed')

            workloadctrl.wait()
            workload = workloadctrl.workload()

            if workload not in drivers:
                logger.debug(f'selection driver for {workload}')
                search_space_ctx: SearchSpaceContext = searchspace.new_search_space_ctx(
                    # this name will be used as uid for the bayesian engine too
                    search_space_ctx_name=f'{event["object"]["metadata"]["name"]}_{workload}',
                    raw_search_space_model=event['object'],
                    workload=workload
                )

                search_space = search_space_ctx.model

                sample_interval = config.WAITING_TIME * config.SAMPLE_SIZE

                production = Instance(name=production_sanitized, namespace=config.NAMESPACE, is_production=True,
                                      sample_interval_in_secs=sample_interval,
                                      ctx=search_space_ctx)

                training = Instance(name=training_sanitized, namespace=config.NAMESPACE, is_production=False,
                                    sample_interval_in_secs=sample_interval,
                                    ctx=search_space_ctx)

                driver = IterationDriver(workload=workload, search_space=search_space,
                                         production=production, training=training,
                                         max_global_iterations=config.NUMBER_ITERATIONS,
                                         max_local_iterations=config.ITERATIONS_BEFORE_REINFORCE,
                                         max_reinforcement_iterations=round(config.ITERATIONS_BEFORE_REINFORCE *
                                                                            config.REINFORCEMENT_RATIO) or 1,
                                         max_probation_iterations=round(config.ITERATIONS_BEFORE_REINFORCE *
                                                                        config.REINFORCEMENT_RATIO) or 1,
                                         sampling_interval=sample_interval,
                                         n_sampling_subintervals=3, logging_subinterval=0.2, fail_fast=config.FAIL_FAST,
                                         uid=uid,
                                         bayesian=config.BAYESIAN,
                                         n_startup_trial=config.N_STARTUP_JOBS,
                                         n_ei_candidates=config.N_EI_CANDIDATES,
                                         seed=config.RANDOM_SEED)

                production.set_initial_configuration(
                    Configuration.running_config(search_space),
                    driver
                )

                drivers[workload] = driver

            driver = drivers[workload]

            # TODO: create a proper error handling for this section
            try:
                driver.production.patch_current_config()
            except Exception:
                logger.exception('fail to update production pod before iteration')

            try:
                driver.training.patch_current_config()
            except Exception:
                logger.exception('fail to update training pod before iteration')

            it: Iteration = next(driver)

            try:
                driver.production.patch_current_config()
            except Exception:
                logger.exception('fail to update production pod after iteration')

            try:
                driver.training.patch_current_config()
            except Exception:
                logger.exception('fail to update training pod after iteration')

            # if isinstance(it, TunedIteration):
            #     try:
            #         if driver.training.active:
            #             driver.production.max_replicas += 1
            #             driver.training.shutdown()
            #     except Exception:
            #         logger.exception('error to delete training replica')

        except Exception:
            logger.exception('error during tuning iteration')
            stoped = True
            for name, driver in drivers.items():
                driver.production.max_replicas += 1
                driver.training.shutdown()

    logger.warning('search space deleted, stoping smarttuning')


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
            microservices = injector.duplicated_dep
            create_contexts(microservices)
            time.sleep(1)
        except Exception:
            logger.exception(f'error in main loop')
            break


# TODO:
# https://github.com/kubernetes-client/python/blob/cef5e9bd10a6d5ca4d9c83da46ccfe2114cdaaf8/examples/notebooks/intro_notebook.ipynb
# repactor injector using this approach

if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('main loop error')
    finally:
        # TODO: proper shutdown the app
        config.shutdown()
