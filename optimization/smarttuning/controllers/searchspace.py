from bayesian import BayesianEngine, BayesianDTO, BayesianChannel
from controllers.searchspacemodel import *

from controllers.k8seventloop import ListToWatch

class SearchSpaceContext:
    def __init__(self, name: str, namespace: str = config.NAMESPACE):
        self.name = name
        self.api = k8s.client.CustomObjectsApi()
        self.namespace = namespace
        self.bayesian_engines = {}

    def observables(self):
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
        manifests = event['object']['spec']['manifests']
        name = event['object']['metadata']['name']

        search_space = {}
        _manifests = []

        for dict_manifest in manifests:
            manifest = ManifestBase(dict_manifest)
            _manifests.append(manifest)
            search_space.update(manifest.get_hyper_interval())

        self.bayesian_engines[self.name] = BayesianEngine(name=self.name, space=search_space, is_bayesian=is_bayesian)

    def delete_bayesian_searchspace(self, event=None):
        # name = event['object']['metadata']['name']
        BayesianChannel.unregister(self.name)
        logger.warning(f'cannot stop engine "{self.name}" -- method not implemented yet')
        return

    def put_into_engine(self, engine_name: str, value: BayesianDTO):
        engine: BayesianEngine = self.bayesian_engines.get(engine_name, None)
        if engine:
            engine.put(value)

    def get_from_engine(self, engine_name: str):
        if engine_name in self.bayesian_engines:
            return self.bayesian_engines[engine_name].get()

    def get_best_so_far(self, engine_name: str):
        if engine_name in self.bayesian_engines:
            return self.bayesian_engines[engine_name].best_so_far()
