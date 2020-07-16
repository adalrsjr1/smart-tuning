from __future__ import annotations
from bayesian import BayesianEngine, BayesianDTO, BayesianChannel
from controllers import k8seventloop
from controllers.searchspacemodel import *

from controllers.k8seventloop import ListToWatch

def init(loop):
    loop.register('searchspaces-controller', ListToWatch(func=k8s.client.CustomObjectsApi().list_namespaced_custom_object,
                                                   namespace=config.NAMESPACE,
                                                   group='smarttuning.ibm.com',
                                                   version='v1alpha1',
                                                   plural='searchspaces'), searchspace_controller)
search_spaces = {}
def searchspace_controller(event):
    t = event.get('type', None)
    if 'ADDED' == t:
        name = event['object']['metadata']['name']
        namespace = event['object']['metadata']['namespace']
        manifests = event['object']['spec']['manifests']
        ctx = SearchSpaceContext(name, namespace, manifests)
        ctx.create_bayesian_searchspace(event, is_bayesian=config.BAYESIAN)
        search_spaces[name] = ctx
    elif 'DELETED' == t:
        name = event['object']['metadata']['name']
        ctx = search_spaces[name]
        ctx.delete_bayesian_searchspace(event)
        del search_spaces[name]

def context(name) -> SearchSpaceContext:
    if name in search_spaces:
        return search_spaces[name]

class SearchSpaceContext:
    def __init__(self, name: str, namespace: str = config.NAMESPACE, manifests = []):
        self.name = name
        self.api = k8s.client.CustomObjectsApi()
        self.namespace = namespace
        self.engine:BayesianEngine = None
        self.manifests = [ManifestBase(manifest) for manifest in manifests]

    def function_of_observables(self):
        api = k8s.client.CustomObjectsApi()
        return ListToWatch(api.list_namespaced_custom_object, namespace=self.namespace,
                           group='smarttuning.ibm.com',
                           version='v1alpha1',
                           plural='searchspaces')

    def selector(self, event):
        t = event.get('type', None)
        if 'ADDED' == t:
            self.create_bayesian_searchspace(event, is_bayesian=config.BAYESIAN)
        elif 'DELETED' == t:
            self.delete_bayesian_searchspace(event, is_bayesian=config.BAYESIAN)

    def create_bayesian_searchspace(self, event, is_bayesian):
        search_spaces_manifests = event['object']['spec']['manifests']

        search_space = {}
        manifests = []

        for dict_manifest in search_spaces_manifests:
            manifest = ManifestBase(dict_manifest)
            manifests.append(manifest)
            search_space.update(manifest.get_hyper_interval())

        self.engine = BayesianEngine(name=self.name, space=search_space, is_bayesian=is_bayesian)

    def delete_bayesian_searchspace(self, event=None):
        # name = event['object']['metadata']['name']
        BayesianChannel.unregister(self.name)
        logger.warning(f'cannot stop engine "{self.name}" -- method not implemented yet')
        return

    def put_into_engine(self, value: BayesianDTO):
        if self.engine:
            self.engine.put(value)

    def get_from_engine(self):
        if self.engine:
            return self.engine.get()

    def get_best_so_far(self, ):
        if self.engine:
            return self.engine.best_so_far()
