import logging
import time

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

def create_context(production_microservice, training_microservice):
    logger.info(f'creating context for ({production_microservice},{training_microservice})')
    production_name = production_microservice
    training_name = training_microservice

    production_sanitized = production_microservice.replace('.', '\\.')
    training_sanitized = training_microservice.replace('.', '\\.')
    logger.info(f'pod_names --  prod:{production_sanitized} train:{training_sanitized}')

    while production_name and training_name:
        # get workloads
        try:
            search_space_ctx: SearchSpaceContext = searchspace.search_spaces.get(f'{production_name}', None)
            if not search_space_ctx:
                return

            if search_space_ctx:
                production = Instance(name=production_sanitized, namespace=config.NAMESPACE, is_production=True,
                                      sample_interval_in_secs=config.WAITING_TIME * config.SAMPLE_SIZE,
                                      ctx=search_space_ctx)
                training = Instance(name=training_sanitized, namespace=config.NAMESPACE, is_production=False,
                                    sample_interval_in_secs=config.WAITING_TIME * config.SAMPLE_SIZE,
                                    ctx=search_space_ctx)
                p = Planner(production=production, training=training, ctx=search_space_ctx,
                            k=config.ITERATIONS_BEFORE_REINFORCE, ratio=config.REINFORCEMENT_RATIO,
                            when_try=config.TRY_BEST_AT_EVERY, restart_trigger=config.RESTART_TRIGGER)

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
