from __future__ import annotations

from threading import Lock

from kubernetes.client.rest import ApiException
from optuna.samplers import TPESampler, RandomSampler

from bayesian import BayesianEngine, BayesianChannel, SmartTuningPrunner
from controllers.injector import duplicate_deployment_for_training
from controllers.k8seventloop import ListToWatch
from controllers.searchspacemodel import *
from models.bayesiandto import BayesianDTO
from models.configuration import Configuration, EmptyConfiguration

logger = logging.getLogger(config.SEARCH_SPACE_LOGGER)
logger.setLevel(logging.DEBUG)


def init(loop):
    loop.register('searchspaces-controller',
                  ListToWatch(func=config.customApi().list_namespaced_custom_object,
                              namespace=config.NAMESPACE,
                              group='smarttuning.ibm.com',
                              version='v1alpha2',
                              plural='searchspaces'), searchspace_controller)


search_spaces: dict[str, dict] = {}
search_space_lock = Lock()


def searchspace_controller(event):
    t = event.get('type', None)
    if 'ADDED' == t:
        try:
            name = event['object']['metadata']['name']
            namespace = event['object']['metadata']['namespace']
            deployment_name = event['object']['spec']['deployment']

            # raw_search_space_model = event['object']
            # ctx = new_search_space_ctx(name, raw_search_space_model)

            deployment = get_deployment(deployment_name, namespace)
            duplicate_deployment_for_training(deployment)
            # update_n_replicas(deployment_name, namespace, deployment.spec.replicas - 1)

            with search_space_lock:
                logger.warning(f'initialiazing search space {name}:{deployment_name}')
                search_spaces[deployment_name] = event

        except ApiException as e:
            if 422 == e.status:
                logger.warning(f'failed to duplicate deployment {name}')
    elif 'DELETED' == t:
        name = event['object']['metadata']['name']
        namespace = event['object']['metadata']['namespace']
        deployment_name = event['object']['spec']['deployment']
        logger.warning(f'removing {name}')
        if deployment_name in search_spaces:
            with search_space_lock:
                search_spaces[deployment_name] = event


def new_search_space_ctx(search_space_ctx_name: str, raw_search_space_model: dict, workload: str) -> SearchSpaceContext:
    sampler = RandomSampler(seed=config.RANDOM_SEED)
    if config.BAYESIAN:
        sampler = TPESampler(
            n_startup_trials=config.N_STARTUP_JOBS,
            n_ei_candidates=config.N_EI_CANDIDATES,
            # gamma=config.GAMMA,
            seed=config.RANDOM_SEED
        )
    study = optuna.create_study(
        sampler=sampler,
        study_name=workload
    )

    ctx = SearchSpaceContext(search_space_ctx_name,
                             SearchSpaceModel(raw_search_space_model, study), workload)
    ctx.create_bayesian_searchspace(
        study,
        max_evals=config.NUMBER_ITERATIONS,
        max_evals_no_change=round(config.MAX_N_ITERATION_NO_IMPROVEMENT),
        workload=workload
    )

    study.pruner = SmartTuningPrunner(workload, ctx)

    return ctx


def stop_tuning(ctx):
    ctx.delete_bayesian_searchspace()
    deployment = get_deployment(ctx.model.deployment, ctx.model.namespace)
    train_deployment = ctx.model.deployment + config.PROXY_TAG
    # if deployment.spec.replicas > 1:
    #     update_n_replicas(ctx.model.deployment, ctx.model.namespace, deployment.spec.replicas + 1)

    logger.debug(f'search spaces names: {search_spaces.keys()}')
    logger.debug(f'search space to delete: {ctx.model.deployment}')
    del search_spaces[ctx.model.deployment]
    config.appsApi().delete_namespaced_deployment(name=train_deployment, namespace=ctx.model.namespace)


def context(deployment_name) -> SearchSpaceContext:
    return search_spaces.get(deployment_name, None)


class SearchSpaceContext:
    def __init__(self, name: str, search_space: SearchSpaceModel, workload: str):
        self.name = name
        self.deployment = search_space.deployment
        self.namespace = search_space.namespace
        self.service = search_space.service
        self.model = search_space
        self.api = config.customApi()
        self.engine: BayesianEngine = None
        self.workload: str = workload

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

    def create_bayesian_searchspace(self, study: optuna.study.Study, max_evals: int, max_evals_no_change: int =100, workload: str = ''):
        self.engine = BayesianEngine(
            name=self.name,
            space=self.model,
            max_evals=max_evals,
            max_evals_no_change=max_evals_no_change,
            workload=workload
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
        logger.warning(f'engine:{self.engine} is_running:{self.engine.is_running()}')
        return EmptyConfiguration()

    def task_done_engine(self):
        if self.engine and self.engine.is_running():
            self.engine.task_done()

    def join_engine(self):
        if self.engine and self.engine.is_running():
            self.engine.join()

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

    # TODO: to remove
    # def get_trials_as_documents(self):
    #     return self.engine.trials_as_documents()
