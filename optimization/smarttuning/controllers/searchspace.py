from __future__ import annotations
from bayesian import BayesianEngine, BayesianDTO, BayesianChannel
from controllers import k8seventloop
from kubernetes.client.models import *
from controllers.searchspacemodel import *
from controllers.k8seventloop import ListToWatch
from controllers.injector import duplicate_deployment_for_training
from kubernetes.client.rest import ApiException

logger = logging.getLogger(config.SEARCH_SPACE_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)

def init(loop):
    loop.register('searchspaces-controller', ListToWatch(func=k8s.client.CustomObjectsApi().list_namespaced_custom_object,
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
            ctx = SearchSpaceContext(name, SearchSpaceModel(event['object']))

            deployment = get_deployment(ctx.model.deployment, ctx.model.namespace)
            duplicate_deployment_for_training(deployment)
            update_n_replicas(ctx.model.deployment, ctx.model.namespace, deployment.spec.replicas - 1)

            ctx.create_bayesian_searchspace(is_bayesian=config.BAYESIAN)
            search_spaces[ctx.model.deployment] = ctx
        except ApiException as e:
            if 422 == e.status:
                logger.warning(f'failed to duplicate deployment {name}')


    elif 'DELETED' == t:
        name = event['object']['spec']['deployment']
        ctx = search_spaces[name]
        ctx.delete_bayesian_searchspace(event)
        deployment = get_deployment(ctx.model.deployment, ctx.model.namespace)
        if deployment.spec.replicas > 1:
            update_n_replicas(ctx.model.deployment, ctx.model.namespace, deployment.spec.replicas + 1)

        del search_spaces[name]

def context(deployment_name) -> SearchSpaceContext:
    return search_spaces.get(deployment_name, None)

class SearchSpaceContext:
    def __init__(self, name: str, search_space:SearchSpaceModel):
        self.name = name
        self.namespace = search_space.namespace
        self.service = search_space.service
        self.model = search_space
        self.api = k8s.client.CustomObjectsApi()
        self.engine:BayesianEngine = None

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
        for manifest in self.model.manifests:
            name = manifest.name
            current_manifest_config = manifest.get_current_config()
            logger.debug(f'getting manifest {name}:{current_manifest_config}')
            current_config[name] = current_manifest_config
        return current_config

    def create_bayesian_searchspace(self, is_bayesian):
        search_space = self.model.search_space()
        self.engine = BayesianEngine(name=self.name, space=search_space, is_bayesian=is_bayesian)

    def delete_bayesian_searchspace(self):
        BayesianChannel.unregister(self.name)
        self.engine.stop()
        logger.warning(f'cannot stop engine "{self.name}" -- method not implemented yet')
        return

    def put_into_engine(self, value: BayesianDTO):
        if self.engine:
            self.engine.put(value)

    def get_from_engine(self):
        if self.engine:
            return self.engine.get()
        return {}

    def get_best_so_far(self, ):
        if self.engine:
            return self.engine.best_so_far()
        return None

    def update_best_loss(self, new_loss:float):
        if self.engine:
            self.engine.update_best_trial(new_loss)

    def get_trials(self):
        if self.engine:
            return self.engine.trials()
        return []

    def get_trials_as_documents(self):
        return self.engine.trials_as_documents()