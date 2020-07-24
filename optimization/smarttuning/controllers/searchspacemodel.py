import copy
import logging
import time

import hyperopt.hp
import hyperopt.pyll.stochastic
import kubernetes as k8s
from hyperopt.pyll.base import scope

import config

logger = logging.getLogger(config.SEARCH_SPACE_LOGGER)
logger.setLevel(logging.DEBUG)


class ManifestBase:
    def __init__(self, manifest):
        self.__dict__.update(manifest)
        manifest_type = self.__dict__['type']
        if 'deployment' in manifest_type and manifest_type['deployment']:
            self.__dict__['type'] = ManifestDeployment(self, self.__dict__['type'])
        else:
            self.__dict__['type'] = ManifestConfigMap(self, self.__dict__['type'])

        if 'params' in self.__dict__:
            params = []
            for param in self.__dict__['params']:
                params.append(BaseParam(param))
            self.__dict__['params'] = params
        else:
            self.__dict__['params'] = []

    def get_hyper_interval(self):
        hyper_params = {}
        for param in self.params:
            hyper_params.update(param.get_hyper_interval())

        return {self.name: hyper_params}

    def patch(self, new_config: dict, production=False):
        return self.type.patch(new_config, production)

    def __repr__(self):
        return str(self.__dict__)


class ManifestDeployment:
    def __init__(self, parent, manifest):
        self.parent = parent
        self.__dict__.update(manifest)

    def __repr__(self):
        return str(self.__dict__)

    def patch(self, new_config: dict, production=False):
        api_instance = k8s.client.AppsV1Api(k8s.client.ApiClient())
        name = self.parent.nameProd if production else self.parent.name
        namespace = self.parent.namespace

        containers = copy.deepcopy(self.containers)
        total_cpu = new_config.get('cpu', None)
        total_memory = new_config.get('memory', None)

        for container in containers:
            container['resources'] = {}
            limits = {}
            if total_cpu:
                limits.update({
                    'cpu': str(float(container['ratio']) * total_cpu)
                })

            if total_memory:
                limits.update({
                    'memory': str(int(float(container['ratio']) * total_memory)) + 'Mi'
                })
            container['resources'] = {'limits': limits, 'requests': limits}
            del (container['ratio'])

        body = {
            "kind": "Deployment",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "spec": {"template": {"spec": {"containers": containers}}}
        }

        return api_instance.patch_namespaced_deployment(name, namespace, body, pretty='true')


class ManifestConfigMap:
    def __init__(self, parent, manifest):
        self.parent = parent
        self.__dict__.update(manifest)

    def __repr__(self):
        return str(self.__dict__)

    def get_configMap(self):
        if False == self.deployment:
            return self.configMap
        return None

    def patch(self, new_config: dict, production=False):
        if False == self.deployment and 'jvm' == self.configMap:
            return self.patch_jvm(new_config, production)
        return self.patch_envvar(new_config, production)

    def patch_envvar(self, new_config, production=False):
        name = self.parent.nameProd if production else self.parent.name
        namespace = self.parent.namespace

        for param in self.parent.params:
            if not param.number['continuous']:
                new_config[param.name] = str(int(new_config[param.name]))

        data = new_config

        # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#patch_namespaced_config_map
        body = {
            "kind": "ConfigMap",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "data": data
        }

        api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())

        return api_instance.patch_namespaced_config_map(name, namespace, body, pretty='true')

    def patch_jvm(self, new_config, production=False):
        name = self.parent.nameProd if production else self.parent.name
        namespace = self.parent.namespace
        filename = self.parent.type.filename
        data = copy.deepcopy(new_config)

        params = dict_to_jvmoptions(data)

        body = {
            "kind": "ConfigMap",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "data": {filename: '\n'.join(params)}
        }

        api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())

        return api_instance.patch_namespaced_config_map(name, namespace, body, pretty='true')

def dict_to_jvmoptions(data):
    params = []
    if '-Xmx' in data:
        params.append('-Xmx' + str(data['-Xmx']) + 'm')
        del (data['-Xmx'])

    if '-Dhttp.keepalive' in data:
        params.append('-Dhttp.keepalive=' + str(data['-Dhttp.keepalive']))
        del (data['-Dhttp.keepalive'])

    if '-Dhttp.maxConnections' in data:
        params.append('-Dhttp.maxConnectionse=' + str(data['-Dhttp.maxConnections']))
        del (data['-Dhttp.maxConnections'])

    if '-Xnojit' in data and data['-Xnojit']:
        params.append('-Xnojit')
        del (data['-Xnojit'])

    if '-Xnoaot' in data and data['-Xnoaot']:
        params.append('-Xnoaot')
        del (data['-Xnoaot'])

    return params + [item for item in data.values()]

class BaseParam:
    def __init__(self, manifest):
        self.__dict__.update(manifest)
        if 'number' in self.__dict__:
            self.__dict__['param'] = NumberParam(self.name, self.number)
            del (self.__dict__['boolean'])
        elif self.__dict__['boolean'] == True:
            self.__dict__['param'] = OptionParam(self.name, self.boolean)
        elif 'options' in self.__dict__:
            self.__dict__['param'] = OptionParam(self.name, self.options)
            del (self.__dict__['boolean'])

    def get_hyper_interval(self):
        return self.param.get_hyper_interval()

    def __repr__(self):
        return str(self.__dict__)


class NumberParam:
    def __init__(self, name, manifest):
        self.name = name
        self.__dict__.update(manifest)
        if not 'step' in self.__dict__:
            self.__dict__['step'] = None

    def get_lower(self):
        if self.continuous:
            return float(self.lower)
        return int(self.lower)

    def get_upper(self):
        if self.continuous:
            return float(self.upper)
        return int(self.upper)

    def get_step(self):
        if self.step is None:
            return None
        if self.continuous:
            return float(self.step)
        return int(self.step)

    def get_hyper_interval(self):
        step = self.get_step()
        to_int = lambda x: scope.int(x) if not self.continuous else x
        if step:
            return {
                self.name: to_int(hyperopt.hp.quniform(self.name, self.get_lower(), self.get_upper(), self.get_step()))}
        else:
            return {self.name: to_int(hyperopt.hp.uniform(self.name, self.get_lower(), self.get_upper()))}

    def __repr__(self):
        return f'(lower: {self.get_lower()}, upper: {self.get_upper()}, step:{self.get_step()})'


class OptionParam:
    def __init__(self, name, manifest):
        self.name = name
        if not isinstance(manifest, bool):
            self.__dict__.update(manifest)
        else:
            self.__dict__.update({'type': 'string', 'values': ['true', 'false']})

    def get_hyper_interval(self):
        return {self.name: hyperopt.hp.choice(self.name, self.values)}

    def __repr__(self):
        return str(self.values)
