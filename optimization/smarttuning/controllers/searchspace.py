from __future__ import annotations

from threading import Lock

from kubernetes.client.rest import ApiException

from controllers.injector import duplicate_deployment_for_training
from controllers.k8seventloop import ListToWatch
from controllers.searchspacemodel import *
from models.workload import Workload, empty_workload

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

            deployment = get_deployment(deployment_name, namespace)
            duplicate_deployment_for_training(deployment)

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


def search_space_model(raw_search_space_model: dict):
    return SearchSpaceModel(raw_search_space_model)


def new_search_space_ctx(search_space_ctx_name: str, raw_search_space_model: dict,
                         workload: Workload) -> SearchSpaceContext:
    ctx = SearchSpaceContext(search_space_ctx_name,
                             SearchSpaceModel(raw_search_space_model), workload)

    return ctx


def context(deployment_name) -> SearchSpaceContext:
    return search_spaces.get(deployment_name, None)


class SearchSpaceContext:
    def __init__(self, name: str, search_space: SearchSpaceModel, workload: Workload):
        self.name = name
        self.deployment = search_space.deployment
        self.namespace = search_space.namespace
        self.service = search_space.service
        self.model = search_space
        self.api = config.customApi()
        self.workload: Workload = workload

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
