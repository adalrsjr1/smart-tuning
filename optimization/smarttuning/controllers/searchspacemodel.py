import copy
import logging
import time
import json
import hyperopt.hp
import hyperopt.pyll.stochastic
import kubernetes as k8s
from kubernetes.client.models import *
from hyperopt.pyll.base import scope
from distutils.util import strtobool
import config

logger = logging.getLogger(config.SEARCH_SPACE_LOGGER)
logger.setLevel(logging.DEBUG)

class SearchSpaceModel:
    def __init__(self, o):
        if o:
            temp =  self.parse_manifests(o)
            self.deployment = temp['deployment']
            self.manifests = temp['models']
            self.namespace = temp['namespace']
            self.service = temp['service']

    def parse_manifests(self, o:dict) -> dict:
        manifests = o['spec']['manifests']
        data = o['data']
        context = o['spec']['deployment']
        namespace = o['spec']['namespace']
        service = o['spec']['service']
        models = []
        for d in data:
            for manifest in manifests:
                if manifest['name'] == d['name']:
                    tunables = d['tunables']
                    name = d['name']
                    filename = d['filename']
                    deployment = manifest['name']
                    manifest_type = manifest['type']

                    if 'deployment' == manifest_type:
                        models.append(DeploymentSearchSpaceModel(name, namespace, deployment, tunables))
                    elif 'configMap' == manifest_type:
                        models.append(ConfigMapSearhSpaceModel(name, namespace, deployment, filename, tunables))
        return {'namespace': namespace, 'service': service, 'deployment': context, 'models': models}

    def search_space(self):
        ss = {}
        for manifest in self.manifests:
            ss[manifest.name] = manifest.search_space()
        return ss

class DeploymentSearchSpaceModel:
    def __init__(self, name, namespace, deployment, tunables):
        self.name = name
        self.namespace = namespace
        self.deployment = deployment
        self.tunables = self.__parse_tunables(tunables)


    def __repr__(self):
        return f'{{"name": "{self.name}", "namespace": "{self.namespace}", "deployment": "{self.deployment}"' \
               f'tunables: {str(self.tunables)}}}'

    def __parse_tunables(self, tunables):
        parameters = {}
        for type_range, list_tunables in tunables.items():
            for r in list_tunables:
                if 'boolean' == type_range:
                    r['values'] = [True, False]
                    parameters[r['name']] = OptionRangeModel(r)
                elif 'number' == type_range:
                    parameters[r['name']] = NumberRangeModel(r)
                elif 'option' == type_range:
                    parameters[r['name']] = OptionRangeModel(r)

        return parameters

    def search_space(self):
        model = {}
        for key, tunable in self.tunables.items():
            model.update(tunable.get_hyper_interval())

        logger.info(f'search space: {model}')
        return model

    def patch(self, new_config: dict, production=False):
        api_instance = config.appsApi()
        name = self.name if production else self.name + config.PROXY_TAG
        namespace = self.namespace

        deployment = get_deployment(name, namespace)
        containers = deployment.spec.template.spec.containers

        total_cpu = new_config.get('cpu', None)
        total_memory = new_config.get('memory', None)

        container: V1Container
        for container in containers:
            resources: V1ResourceRequirements = container.resources

            if not resources:
                resources = V1ResourceRequirements(limits={}, requests={})

            limits = resources.limits
            if not limits:
                limits = {}

            if total_cpu:
                limits.update({
                    'cpu': str(total_cpu / len(containers))
                })

            if total_memory:
                limits.update({
                    'memory': str(int(total_memory / len(containers))) + 'Mi'
                })

            container.resources.limits = limits

        body = {
            "kind": "Deployment",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "spec": {"template": {"spec": {"containers": containers}}}
        }

        if 'replicas' in new_config:
            body['spec'].update({'replicas': new_config['replicas']})


        return api_instance.patch_namespaced_deployment(name, namespace, body, pretty='true')

def update_n_replicas(deployment_name, namespace, curr_n_replicas):
    if curr_n_replicas > 1:
        logger.info(f'updating {deployment_name} replicas to {curr_n_replicas}')
        body = {"spec":{"replicas":curr_n_replicas}}
        config.appsApi().patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=body)

def get_deployment(name, namespace) -> V1Deployment:
    return config.appsApi().read_namespaced_deployment(name, namespace)

class ConfigMapSearhSpaceModel:
    def __init__(self, name, namespace, deployment, filename, tunables):
        self.name = name
        self.namespace = namespace
        self.deployment = deployment
        self.filename = filename
        self.tunables = self.__parse_tunables(tunables)

    def __repr__(self):
        return f'{{"name": "{self.name}", "namespace": "{self.namespace}", "deployment": "{self.deployment}"' \
               f'tunables: {str(self.tunables)}}}'

    def __parse_tunables(self, tunables):
        parameters = {}
        for type_range, list_tunables in tunables.items():
            for r in list_tunables:
                if 'boolean' == type_range:
                    r['values'] = [True, False]
                    parameters[r['name']] = OptionRangeModel(r)
                elif 'number' == type_range:
                    parameters[r['name']] = NumberRangeModel(r)
                elif 'option' == type_range:
                    parameters[r['name']] = OptionRangeModel(r)

        return parameters

    def search_space(self):
        model = {}
        for key, tunable in self.tunables.items():
            model.update(tunable.get_hyper_interval())

        return model

    def patch(self, new_config: dict, production=False):
        if self.filename == 'jvm.options':
            return self.patch_jvm(new_config, production)
        elif self.filename == '':
            return self.patch_envvar(new_config, production)

    def patch_envvar(self, new_config, production=False):
        name = self.name if production else self.name + config.PROXY_TAG
        namespace = self.namespace

        for key, param in self.tunables.items():
            if isinstance(param, NumberRangeModel):
                if param.get_real():
                    new_config[key] = str(float(new_config[key]))
                else:
                    new_config[key] = str(int(new_config[key]))
            else:
                new_config[key] = str(new_config[key])

        data = new_config

        # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#patch_namespaced_config_map
        body = {
            "kind": "ConfigMap",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "data": data
        }

        return config.coreApi().patch_namespaced_config_map(name, namespace, body, pretty='true')

    def patch_jvm(self, new_config, production=False):
        name = self.name if production else self.name + config.PROXY_TAG
        namespace = self.namespace
        filename = self.filename

        data = copy.deepcopy(new_config)
        params = dict_to_jvmoptions(data)
        body = {
            "kind": "ConfigMap",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "data": {filename: '\n'.join(params)}
        }

        return config.coreApi().patch_namespaced_config_map(name, namespace, body, pretty='true')


class NumberRangeModel:
    def __init__(self, r):
        self.name = r['name']
        self.upper = r['upper']
        self.lower = r['lower']
        self.step = r.get('step', 0)
        self.real = r.get('real', False)

    def __repr__(self):
        return f'{{"name": "{self.name}", "lower": {self.get_lower()}, "upper": {self.get_upper()}, "real": "{self.get_real()}", "step": {self.get_step()}}}'

    def get_upper(self):
        return float(self.upper)

    def get_lower(self):
        return float(self.lower)

    def get_step(self):
        return float(self.step)

    def get_real(self):
        if isinstance(self.real, bool):
            return self.real
        return strtobool(self.real)

    def get_hyper_interval(self):
        to_int = lambda x: x if self.get_real() else scope.int(x)

        upper = self.get_upper()
        if self.get_lower() == upper:
            upper += 0.1

        logger.debug(f'{self}')
        if self.get_step():
            return {
                self.name: to_int(hyperopt.hp.quniform(self.name, self.get_lower(), self.get_upper(), self.get_step()))}
        else:
            return {self.name: to_int(hyperopt.hp.uniform(self.name, self.get_lower(), self.get_upper()))}


class OptionRangeModel:
    def __init__(self, r):
        self.name = r['name']
        self.values = r['values']

    def __repr__(self):
        return f'{{"name": "{self.name}, "values": {json.dumps(self.values)} }}'

    def get_hyper_interval(self):
        return {self.name: hyperopt.hp.choice(self.name, self.values)}


def dict_to_jvmoptions(data):
    params = []
    if '-Xmx' in data:
        params.append('-Xmx' + str(data['-Xmx']) + 'm')
        del (data['-Xmx'])

    if '-Dhttp.keepalive' in data:
        params.append('-Dhttp.keepalive=' + str(data['-Dhttp.keepalive']))
        del (data['-Dhttp.keepalive'])

    if '-Dhttp.maxConnections' in data:
        params.append('-Dhttp.maxConnections=' + str(data['-Dhttp.maxConnections']))
        del (data['-Dhttp.maxConnections'])

    if '-Xtune:virtualized' in data:
        params.append('-Xtune:virtualized')
        del (data['-Xtune:virtualized'])

    if '-Xnojit' in data:
        if data['-Xnojit']:
            params.append('-Xnojit')
        del (data['-Xnojit'])

    if '-Xnoaot' in data:
        if data['-Xnoaot']:
            params.append('-Xnoaot')
        del (data['-Xnoaot'])

    return params + [item for item in data.values()]
