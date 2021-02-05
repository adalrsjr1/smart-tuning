from __future__ import annotations

import optuna
from kubernetes.client.rest import ApiException
from optuna.samplers import TPESampler, RandomSampler

from bayesian import BayesianEngine, BayesianDTO, BayesianChannel
from controllers.injector import duplicate_deployment_for_training
from controllers.k8seventloop import ListToWatch
from controllers.searchspacemodel import *
from models.configuration import Configuration, EmptyConfiguration

logger = logging.getLogger(config.SEARCH_SPACE_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)


def init(loop):
    loop.register('searchspaces-controller',
                  ListToWatch(func=config.customApi().list_namespaced_custom_object,
                              namespace=config.NAMESPACE,
                              group='smarttuning.ibm.com',
                              version='v1alpha2',
                              plural='searchspaces'), searchspace_controller)


search_spaces = {}


def searchspace_controller(event):
    t = event.get('type', None)
    if 'ADDED' == t:
        try:
            name = event['object']['metadata']['name']
            sampler = RandomSampler(seed=config.RANDOM_SEED)
            if config.BAYESIAN:
                sampler = TPESampler(
                    n_startup_trials=config.N_STARTUP_JOBS,
                    n_ei_candidates=config.N_EI_CANDIDATES,
                    # gamma=config.GAMMA,
                    seed=config.RANDOM_SEED
                )
            study = optuna.create_study(sampler=sampler)
            ctx = SearchSpaceContext(name, SearchSpaceModel(event['object'], study))

            deployment = get_deployment(ctx.model.deployment, ctx.model.namespace)
            duplicate_deployment_for_training(deployment)
            update_n_replicas(ctx.model.deployment, ctx.model.namespace, deployment.spec.replicas - 1)

            ctx.create_bayesian_searchspace(study, max_evals=config.NUMBER_ITERATIONS)
            search_spaces[ctx.model.deployment] = ctx
        except ApiException as e:
            if 422 == e.status:
                logger.warning(f'failed to duplicate deployment {name}')
    elif 'DELETED' == t:
        name = event['object']['spec']['deployment']
        if name in search_spaces:
            ctx = search_spaces[name]
            stop_tuning(ctx)


def stop_tuning(ctx):
    ctx.delete_bayesian_searchspace()
    deployment = get_deployment(ctx.model.deployment, ctx.model.namespace)
    train_deployment = ctx.model.deployment + config.PROXY_TAG
    if deployment.spec.replicas > 1:
        update_n_replicas(ctx.model.deployment, ctx.model.namespace, deployment.spec.replicas + 1)

    logger.debug(f'search spaces names: {search_spaces.keys()}')
    logger.debug(f'search space to delete: {ctx.model.deployment}')
    del search_spaces[ctx.model.deployment]
    config.appsApi().delete_namespaced_deployment(name=train_deployment, namespace=ctx.model.namespace)


def context(deployment_name) -> SearchSpaceContext:
    return search_spaces.get(deployment_name, None)


class SearchSpaceContext:
    def __init__(self, name: str, search_space: SearchSpaceModel):
        self.name = name
        self.namespace = search_space.namespace
        self.service = search_space.service
        self.model = search_space
        self.api = config.customApi()
        self.engine: BayesianEngine = None

    def function_of_observables(self):
        return ListToWatch(config.coreApi().list_namespaced_custom_object, namespace=self.namespace,
                           group='smarttuning.ibm.com',
                           version='v1alpha2',
                           plural='searchspaces')

    def selector(self, event):
        t = event.get('type', None)
        if 'ADDED' == t:
            self.create_bayesian_searchspace(is_bayesian=config.BAYESIAN)
        elif 'DELETED' == t:
            self.delete_bayesian_searchspace(is_bayesian=config.BAYESIAN)

    def get_current_config(self):
        logger.info('getting current config')
        current_config = {}
        manifest: BaseSearchSpaceModel
        for manifest in self.model.manifests:
            name = manifest.name
            current_config[name] = manifest.get_current_config()
            logger.debug(f'getting manifest {name}:{current_config[name]}')

        return current_config

    def create_bayesian_searchspace(self, study: optuna.study.Study, max_evals: int):
        self.engine = BayesianEngine(
            name=self.name,
            space=self.model,
            max_evals=max_evals
        )

    def delete_bayesian_searchspace(self):
        self.engine.stop()
        BayesianChannel.unregister(self.name)

    def put_into_engine(self, value: BayesianDTO):
        if self.engine and self.engine.is_running():
            self.engine.put(value)

    def get_from_engine(self) -> Configuration:
        if self.engine and self.engine.is_running():
            return self.engine.get()
        return EmptyConfiguration()

    def get_best_so_far(self, ):
        if self.engine:
            return self.engine.best_so_far()
        return None

    def update_best_loss(self, new_loss: float):
        if self.engine:
            self.engine.update_best_trial(new_loss)

    def get_trials(self):
        if self.engine:
            return self.engine.trials()
        return []

    def get_smarttuning_trials(self):
        if self.engine:
            return self.engine.smarttuning_trials
        return None

    def get_trials_as_documents(self):
        return self.engine.trials_as_documents()
