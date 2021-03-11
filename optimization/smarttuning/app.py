import logging
import time
from threading import Condition

import config
from controllers import injector, searchspace
from controllers.k8seventloop import EventLoop
from controllers.planner import Planner
from controllers.searchspace import SearchSpaceContext
from models.configuration import Configuration
from models.instance import Instance

logger = logging.getLogger(config.APP_LOGGER)
logger.setLevel('DEBUG')


def init():
    event_loop = EventLoop(config.executor())
    # initializing controllers
    searchspace.init(event_loop)
    injector.init(event_loop)


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

    # TODO: need further improvements
    # to_remove = []
    # for microservice, future in contexts.items():
    #     if future.done():
    #         logger.debug(f'gc: marking to remove {microservice} ctx')
    #         to_remove.append(microservice)
    #
    # for microservice in to_remove:
    #     logger.debug(f'gc contexts: wiping {microservice} ctx')
    #     del contexts[microservice]
    #
    # if 0 == len(microservices) < len(contexts):
    #     future:Future
    #     for future in contexts.values():
    #         future.cancel()


class SmartTuningContext:
    def __init__(self,
                 search_space_ctx: SearchSpaceContext,
                 production: Instance,
                 training: Instance,
                 planner: Planner,
                 workload: str = ''):
        self._search_space_ctx: SearchSpaceContext = search_space_ctx
        self._production: Instance = production
        self._training: Instance = training
        self._planner: Planner = planner
        self._workload: str = workload

    @property
    def search_space_ctx(self):
        return self.search_space_ctx

    @property
    def production(self):
        return self._production

    @property
    def training(self):
        return self._training

    @property
    def planner(self):
        return self._planner

    @property
    def workload(self):
        return self._workload

    def progress(self) -> (Configuration, bool):
        configuration, is_last_iteration = next(self.planner)
        return configuration, is_last_iteration

    def stop(self):
        pass


def curr_workload() -> str:
    return ''


def create_context(production_microservice, training_microservice):
    logger.info(f'creating context for ({production_microservice},{training_microservice})')
    production_name = production_microservice
    training_name = training_microservice

    production_sanitized = production_microservice.replace('.', '\\.')
    training_sanitized = training_microservice.replace('.', '\\.')
    logger.info(f'pod_names --  prod:{production_sanitized} train:{training_sanitized}')

    context_by_workload: list[SmartTuningContext] = []

    workload = None
    counter = 0
    while counter < config.NUMBER_ITERATIONS:
        try:
            # get event from search space deployment
            with searchspace.search_space_lock:
                event = searchspace.search_spaces.get(production_name)

            # busy waiting if search space wasn't deployed yet
            if not event:
                logger.info(f'{production_name} waiting for search space')
                time.sleep(0.1)
                continue

            # check if there is a context for the current workload, if not, create a new context
            if len(context_by_workload) > 0 and workload == curr_workload():
                smarttuning_context = [ctx for ctx in context_by_workload if ctx.workload == workload][0]
            else:
                workload = curr_workload()
                search_space_ctx: SearchSpaceContext = searchspace.new_search_space_ctx(
                    # this name will be used as uid for the bayesian engine too
                    search_space_ctx_name=f'{event["object"]["metadata"]["name"]}_{workload}',
                    raw_search_space_model=event['object']
                )

                production = Instance(name=production_sanitized, namespace=config.NAMESPACE, is_production=True,
                                      sample_interval_in_secs=config.WAITING_TIME * config.SAMPLE_SIZE,
                                      ctx=search_space_ctx)
                training = Instance(name=training_sanitized, namespace=config.NAMESPACE, is_production=False,
                                    sample_interval_in_secs=config.WAITING_TIME * config.SAMPLE_SIZE,
                                    ctx=search_space_ctx)
                p = Planner(production=production, training=training, ctx=search_space_ctx,
                            k=config.ITERATIONS_BEFORE_REINFORCE, ratio=config.REINFORCEMENT_RATIO,
                            when_try=config.TRY_BEST_AT_EVERY, restart_trigger=config.RESTART_TRIGGER)

                smarttuning_context = SmartTuningContext(
                    search_space_ctx=search_space_ctx,
                    production=production,
                    training=training,
                    planner=p,
                    workload=workload
                )
                context_by_workload.append(smarttuning_context)
                print('XXXXX ', context_by_workload, len(context_by_workload))

            # iterate trying a new config
            configuration, last_iteration = smarttuning_context.progress()
            logger.info(f'[{counter}, last:{last_iteration}] {configuration}')
            counter += 1

        except:
            logger.exception('error during tuning iteration')

    del training
    # logger.info(f'stopping tuning for {production_name}:: {context_by_workload}')
    for ctx in context_by_workload:
        searchspace.stop_tuning(ctx.search_space_ctx)




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
        except:
            logger.exception(f'error in main loop')
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
