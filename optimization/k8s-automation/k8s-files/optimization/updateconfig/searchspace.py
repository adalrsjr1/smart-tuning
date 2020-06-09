import copy
import os
import time
from pprint import pprint

import hyperopt.hp
import hyperopt.pyll.stochastic
import kubernetes as k8s

from updateconfig import bayesian


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
        self.type.patch(new_config, production)

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
        pretty = 'true'

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

        return api_instance.patch_namespaced_deployment(name, namespace, body)


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
        pretty = 'true'

        return api_instance.patch_namespaced_config_map(name, namespace, body, pretty=pretty)

    def patch_jvm(self, new_config, production=False):
        name = self.parent.nameProd if production else self.parent.name
        namespace = self.parent.namespace
        filename = self.parent.type.filename
        data = new_config

        params = [item for item in new_config.values()]
        if '-Xmx' in data:
            params.append('-Xmx' + str(data['-Xmx']) + 'm')

        if '-Xnojit' in data and data['-Xnojit']:
            params.append('-Xnojit')

        if '-Xnoaot' in data and data['-Xnoaot']:
            params.append('-Xnoaot')

        body = {
            "kind": "ConfigMap",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "data": {filename: '\n'.join(params)}
        }

        api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())
        pretty = 'true'

        return api_instance.patch_namespaced_config_map(name, namespace, body, pretty=pretty)


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
        if step:
            return {self.name: hyperopt.hp.quniform(self.name, self.get_lower(), self.get_upper(), self.get_step())}
        else:
            return {self.name: hyperopt.hp.uniform(self.name, self.get_lower(), self.get_upper())}

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


def init(cdr_search_space_name, namespace):
    # init k8s config

    if 'KUBERNETES_SERVICE_HOST' in os.environ:
        k8s.config.load_incluster_config()
    else:
        k8s.config.load_kube_config()

    # load CRD search space
    api = k8s.client.CustomObjectsApi()
    search_space_manifest = api.get_namespaced_custom_object(name=cdr_search_space_name, namespace=namespace,
                                                             group='smarttuning.ibm.com', version='v1alpha1',
                                                             plural='searchspaces')

    # parse dicts to objects
    manifests = search_space_manifest['spec']['manifests']
    search_space = {}
    _manifests = []

    for dict_manifest in manifests:
        manifest = ManifestBase(dict_manifest)
        _manifests.append(manifest)
        search_space.update(manifest.get_hyper_interval())

    return _manifests, search_space


if __name__ == '__main__':

    import random


    class Mock:
        def objective(self):
            return random.randint(10, 100)


    manifests, search_space = init(cdr_search_space_name='acmeair-searchspace', namespace='default')
    bayesian.init(search_space)
    bayesian.put(Mock())

    # update manifests
    configuration = list(bayesian.get().items())
    print('>>>', dict(configuration).items())
    # for key, value in configuration:
    #     for manifest in manifests:
    #         if key == manifest.name:
    #             manifest.patch(value)
