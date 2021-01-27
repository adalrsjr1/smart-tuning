import copy
import logging
import time
import json
import hyperopt.hp
import hyperopt.pyll.stochastic
import kubernetes as k8s
from kubernetes.client.models import *
from kubernetes.utils import quantity
from hyperopt.pyll.base import scope
from distutils.util import strtobool
import config
import sys

logger = logging.getLogger(config.SEARCH_SPACE_LOGGER)
logger.setLevel(logging.DEBUG)


class SearchSpaceModel:
    def __init__(self, o):
        if o:
            temp = self.parse_manifests(o)

            self.deployment = temp['deployment']
            self.manifests = temp['models']
            self.namespace = temp['namespace']
            self.service = temp['service']

    def parse_manifests(self, o: dict) -> dict:
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

    def search_space(self) -> dict:
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
                    r['values'] = ['True', 'False']
                    r['type'] = 'bool'
                    parameters[r['name']] = OptionRangeModel(r)
                elif 'number' == type_range:
                    parameters[r['name']] = NumberRangeModel(r)
                elif 'option' == type_range:
                    parameters[r['name']] = OptionRangeModel(r)

        return parameters

    def search_space(self):
        model = {}
        for key, tunable in self.tunables.items():
            model.update(tunable.get_hyper_interval(self.tunables))

        logger.info(f'search space: {model}')
        return model

    def get_current_config(self) -> (dict, dict):
        api_instance = config.appsApi()
        deployment: V1Deployment = api_instance.read_namespaced_deployment(name=self.name, namespace=self.namespace)

        return self.get_current_config_from_spec(deployment)

    def get_current_config_from_spec(self, deployment: V1Deployment) -> (dict, dict):
        containers = deployment.spec.template.spec.containers
        config_limits = {'cpu': 0, 'memory': 0}
        container: V1Container
        for container in containers:
            resources: V1ResourceRequirements = container.resources

            if not resources:
                resources = V1ResourceRequirements(limits={}, requests={})

            limits = resources.limits
            if not limits:
                limits = {}

            config_limits['cpu'] += float(quantity.parse_quantity(limits.get('cpu', 0)))
            config_limits['memory'] += float(quantity.parse_quantity(limits.get('memory', 0))) / (2 ** 20)

        keys = set(config_limits.keys()).intersection(set(self.tunables.keys()))

        valued_default_config = {}
        indexed_default_config = {}
        # if Option set index to default config
        # otherwise set limit values
        for k, v in config_limits.items():
            if k in keys and k in self.tunables:
                valued_default_config[k] = v
                if isinstance(self.tunables[k], OptionRangeModel):
                    index_of_value = self.tunables[k].index_of_value(v)
                    if index_of_value < 0:
                        # in not exists set first option
                        indexed_default_config[k] = 0
                    else:
                        indexed_default_config[k] = index_of_value
                else:
                    indexed_default_config[k] = v

        return indexed_default_config, valued_default_config

    def patch(self, new_config: dict, production=False):
        api_instance = config.appsApi()
        name, namespace, containers = self.core_patch(new_config, production)

        body = {
            "kind": "Deployment",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "spec": {"template": {"spec": {"containers": containers}}}
        }

        if 'replicas' in new_config:
            body['spec'].update({'replicas': new_config['replicas']})

        return api_instance.patch_namespaced_deployment(name, namespace, body, pretty='false')

    def core_patch(self, new_config: dict, production=False):
        name = self.name if production else self.name + config.PROXY_TAG
        namespace = self.namespace

        deployment = get_deployment(name, namespace)
        containers = deployment.spec.template.spec.containers

        total_cpu = new_config.get('cpu', None)
        if total_cpu == 0:
            total_cpu = None
        total_memory = new_config.get('memory', None)
        if total_memory == 0:
            total_memory = None

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

        return name, namespace, containers


def update_n_replicas(deployment_name, namespace, curr_n_replicas):
    if curr_n_replicas > 1:
        logger.info(f'updating {deployment_name} replicas to {curr_n_replicas}')
        body = {"spec": {"replicas": curr_n_replicas}}
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
                    r['values'] = ['True', 'False']
                    r['type'] = 'bool'
                    parameters[r['name']] = OptionRangeModel(r)
                elif 'number' == type_range:
                    parameters[r['name']] = NumberRangeModel(r)
                elif 'option' == type_range:
                    parameters[r['name']] = OptionRangeModel(r)

        return parameters

    def search_space(self):
        model = {}
        for key, tunable in self.tunables.items():
            model.update(tunable.get_hyper_interval(self.tunables))

        return model

    def get_current_config(self) -> (dict, dict):
        coreApi = config.coreApi()
        configMap: V1ConfigMap = coreApi.read_namespaced_config_map(name=self.name, namespace=self.namespace)

        return self.get_current_config_core(configMap)

    def get_current_config_core(self, configMap: V1ConfigMap) -> (dict, dict):
        keys = set(configMap.data.keys()).intersection(set(self.tunables.keys()))

        valued_default_config = {}
        indexed_default_config = {}
        # if Option set index to default config
        # otherwise set values
        print(configMap.data.items())
        if 'jvm.options' in configMap.data:
            # TODO: workaround to get inital config from jvm
            if len(configMap.data['jvm.options']) == 0:
                raise RuntimeError(f'config file: {configMap.data} cannot be empty -- set default config')
            else:
                keys, configMap.data = jmvoptions_to_dict(configMap.data['jvm.options'])

        for k, v in configMap.data.items():

            if k in keys and k in self.tunables:
                valued_default_config[k] = v
                if isinstance(self.tunables[k], OptionRangeModel):
                    index_of_value = self.tunables[k].index_of_value(v)
                    if index_of_value < 0:
                        # in not exists set first option
                        indexed_default_config[k] = 0
                    else:
                        indexed_default_config[k] = index_of_value
                else:
                    indexed_default_config[k] = v

        return indexed_default_config, valued_default_config

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

        return config.coreApi().patch_namespaced_config_map(name, namespace, body, pretty='false')

    def core_patch_jvm(self, new_config, production=False):
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
        return name, namespace, body

    def patch_jvm(self, new_config, production=False):
        name, namespace, body = self.core_patch_jvm(new_config, production)

        return config.coreApi().patch_namespaced_config_map(name, namespace, body, pretty='false')


class NumberRangeModel:
    def __init__(self, r):
        self.name = r['name']
        self.upper, self.upper_dep = self.__unpack_value(r['upper'])
        self.lower, self.lower_dep = self.__unpack_value(r['lower'])
        self.step = r.get('step', 0)
        self.real = r.get('real', False)
        ## singleton
        self.__hyper_interval = None

    def __unpack_value(self, item):
        if item and isinstance(item, dict):
            return item['value'], item['dependsOn']
        raise ValueError(
            f'{self.name} has a invalid item:{item}, expecting a dict:{{"dependsOn":"str", "value":"int-or-str"}}')

    def __repr__(self):
        return f'{{"name": "{self.name}", ' \
               f'"lower": {self.get_lower()}, ' \
               f'"lower_dep":{self.get_lower_dep()}, ' \
               f'"upper": {self.get_upper()}, ' \
               f'"upper_dep":{self.get_upper_dep()}, ' \
               f'"real": "{self.get_real()}", ' \
               f'"step": {self.get_step()}}}'

    def __is_string(self, value):
        return isinstance(value, str) and str.isalpha(value)

    def get_upper(self):
        if self.__is_string(self.upper):
            return self.upper
        return float(self.upper)

    def get_upper_dep(self):
        return self.upper_dep

    def get_lower(self):
        if self.__is_string(self.lower):
            return self.lower
        return float(self.lower)

    def get_lower_dep(self):
        return self.lower_dep

    def get_step(self):
        return float(self.step)

    def get_real(self):
        if isinstance(self.real, bool):
            return self.real
        return strtobool(self.real)

    def get_hyper_interval(self, ctx={}) -> dict:
        ## making the interval singleton
        if self.__hyper_interval is not None:
            return self.__hyper_interval
        """ ctx['name'] = 'NumberRangeModel'"""
        to_int = lambda x: x if self.get_real() else scope.int(x)

        upper = self.get_upper()
        if self.get_lower() == upper:
            upper += 0.1

        logger.debug(f'{self}')

        # dep_eval(self.get_upper_dep(), ctx=ctx if self.get_upper_dep() else )
        import numpy as np
        from hyperopt.pyll.stochastic import sample
        inner_ctx = {}
        for k, v in ctx.items():
            item = v.get_hyper_interval()
            inner_ctx.update(item)

        upper = dep_eval(self.get_upper_dep(), ctx=inner_ctx, default=self.get_upper())
        lower = dep_eval(self.get_lower_dep(), ctx=inner_ctx, default=self.get_lower())

    # upper_dep = ctx.get(self.get_upper_dep(), None)
        # lower_dep = ctx.get(self.get_lower_dep(), None)
        #
        # upper = list(upper_dep.get_hyper_interval().values())[0] if upper_dep else self.get_upper()
        # print(upper)
        # lower = list(lower_dep.get_hyper_interval().values())[0] if lower_dep else self.get_lower()
        # print(lower)

        ## begin -- using linear transformation
        if self.get_step():
            value = hyperopt.hp.quniform(self.name, self.get_lower(), self.get_upper(), self.get_step())
        else:
            value = hyperopt.hp.uniform(self.name, self.get_lower(), self.get_upper())

        value = to_scale(
            self.get_lower(),
            lower,
            self.get_upper(),
            upper,
            value
        )
        ## end
        self.__hyper_interval = {self.name: to_int(value)}
        return self.__hyper_interval


def dep_eval(expr, ctx=None, default=None):
    if ctx is None:
        ctx = {}
    if default is None:
        default = {}

    expr_lst = tokenize(expr, ctx, default)
    return polish_eval(expr_lst)


def tokenize(expr, ctx=None, default=None):
    if ctx is None:
        ctx = {}
    if default is None:
        default = {}

    if len(expr) != 0:
        return [dep_eval_expr(token, ctx, default) for token in expr.split()]
    return [dep_eval_expr('', ctx, default)]


def dep_eval_expr(token, ctx=None, default=None):
    if ctx is None:
        ctx = {}
    if default is None:
        default = {}
    import re

    if len(token) == 0:
        if isinstance(default, dict) and len(default) > 0:
            return list(default.values()[0])
        if not isinstance(default, dict):
            return default

    if re.compile(r"[-+]?\d*\.\d+|\d+").match(token):
        return float(token)
    else:
        if token in '+-*/':
            return token
        else:
            if token in ctx:
                return ctx[token]
            if isinstance(default, dict):
                if token in default:
                    return default[token]
                else:
                    raise KeyError(f'cannot evaluate token:{token}, it is not within the ctx')
            elif default is None:
                raise KeyError(f'there is no default value for token:{token}')
            else:
                return default

def polish_eval(expr: list):
    if len(expr) == 0:
        return 0
    if len(expr) == 1:
        if isinstance(expr[0], dict):
            return list(expr[0].values())[0]
        else:
            return expr[0] if str(expr[0]) not in '+-*/' else None
    stack = []

    def op(a, b, s):
        if '+' == s:
            return a + b
        if '-' == s:
            return a - b
        if '*' == s:
            return a * b
        if '/' == s:
            return a / b

    for token in expr:
        if str(token) not in '+-*/':
            stack.insert(0, token)
        else:
            e2 = stack.pop(0)
            e1 = stack.pop(0)
            stack.insert(0, op(e1, e2, token))

    return stack[0]


def to_scale(x1, y1, x2, y2, k):
    """
    does a linear transformation leading the xs values to ys values
    xs are the original limits of a parameter
    ys are the new limits based on their dependencies
    k is the value sampled from the xs interval that will be scaled to ys interval

    for example:
        ...
        - name: "a"
          lower:
            value: 100
          upper:
            value: 200
        - name: "b"
          lower:
            value: 100
            dependsOn: "a"
          upper:
            value: 200
        ...
    the interval b = (100, 200) will be shrink to b = (a, 200), where 'a' is a value sampled from a = (100, 200)
    """
    if x1 == y1 and x2 == y2:
        return k

    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    # print(f'{m}*x+{b} --> p=({x1},{y1}) q=({x2},{y2}) ')
    return (m * k) + b


def normalize(k, mx, mn):
    return (k - mn) / (mx - mn)


class OptionRangeModel:
    def __init__(self, r):
        self.name = r['name']
        self.type = r['type']
        self.values = r['values']

    def __repr__(self):
        return f'{{"name": "{self.name}, "type":{self.type}, "values": {json.dumps(self.values)} }}'

    def index_of_value(self, value):
        if self.type == 'integer':
            value = str(int(value))
        elif self.type == 'real':
            value = str(float(value))
        else:
            value = str(value)

        if value in self.values:
            return self.values.index(value)

        return -1

    def cast(self, values, new_type):
        to_bool = lambda x: bool(strtobool(x))
        types = {'integer': int, 'real': float, 'string': str, 'bool': to_bool}
        return [types[new_type](value) for value in values]

    def get_values(self):
        return self.cast(self.values, self.type)

    def get_hyper_interval(self, ctx={}) -> dict:
        """ ctx['name'] = 'OptionRangeModel'"""
        return {self.name: hyperopt.hp.choice(self.name, self.get_values())}


def jmvoptions_to_dict(jvm_options):
    data = jvm_options.split('\n')
    data = [item for item in data if not item.startswith('#')]
    params = {}
    keys = []
    # if '-Xmx' in data:
    #     params['-Xmx'] = data['']
    #     params.append('-Xmx' + str(data['-Xmx']) + 'm')
    #     del (data['-Xmx'])
    #
    # if '-Dhttp.keepalive' in data:
    #     params.append('-Dhttp.keepalive=' + str(data['-Dhttp.keepalive']))
    #     del (data['-Dhttp.keepalive'])
    #
    # if '-Dhttp.maxConnections' in data:
    #     params.append('-Dhttp.maxConnections=' + str(data['-Dhttp.maxConnections']))
    #     del (data['-Dhttp.maxConnections'])

    params['-Xtune:virtualized'] = False
    params['gc'] = '-Xgcpolicy:gencon'
    params['container_support'] = '-XX:+UseContainerSupport'
    for item in data:
        if item.startswith('-XX:InitialRAMPercentage'):
            params['-XX:InitialRAMPercentage'] = int(item.split('-XX:InitialRAMPercentage=')[1])
        elif item.startswith('-XX:MaxRAMPercentage'):
            params['-XX:MaxRAMPercentage'] = int(item.split('-XX:MaxRAMPercentage=')[1])
        elif item.startswith('-Xmn'):
            params['-Xmn'] = int(item.split('-Xmn')[1].split('m')[0])
        elif item.startswith('-Xms'):
            params['-Xms'] = int(item.split('-Xms')[1].split('m')[0])
        elif item.startswith('-Xmx'):
            params['-Xmx'] = int(item.split('-Xmx')[1].split('m')[0])
        elif item.startswith('-XX:SharedCacheHardLimit'):
            params['-XX:SharedCacheHardLimit'] = int(item.split('-XX:SharedCacheHardLimit=')[1].split('m')[0])
        elif item.startswith('-Xscmx'):
            params['-Xscmx'] = int(item.split('-Xscmx=')[1].split('m')[0])
        elif item.startswith('-Xtune:virtualized'):
            params['-Xtune:virtualized'] = True
        elif item.startswith('-XX:-UseContainerSupport'):
            params['container_support'] = '+XX:-UseContainerSupport'
        elif item.startswith('-XX:+UseContainerSupport'):
            params['container_support'] = '-XX:+UseContainerSupport'
        elif item.startswith('-Xgcpolicy:'):
            params['gc'] = item

    # if '-Xnojit' in data:
    #     if data['-Xnojit']:
    #         params.append('-Xnojit')
    #     del (data['-Xnojit'])
    #
    # if '-Xnoaot' in data:
    #     if data['-Xnoaot']:
    #         params.append('-Xnoaot')
    #     del (data['-Xnoaot'])
    return set(params.keys()), params


def dict_to_jvmoptions(data):
    params = []
    if '-XX:InitialRAMPercentage' in data:
        params.append('-XX:InitialRAMPercentage=' + str(data['-XX:InitialRAMPercentage']))
        del data['-XX:InitialRAMPercentage']

    if '-XX:MaxRAMPercentage' in data:
        params.append('-XX:MaxRAMPercentage=' + str(data['-XX:MaxRAMPercentage']))
        del data['-XX:MaxRAMPercentage']

    if '-Xmn' in data:
        params.append('-Xmn' + str(data['-Xmn']) + 'm')
        del data['-Xmn']

    if '-Xmx' in data:
        params.append('-Xmx' + str(data['-Xmx']) + 'm')
        del data['-Xmx']

    if '-Xms' in data:
        params.append('-Xms' + str(data['-Xms']) + 'm')
        del data['-Xms']

    if '-XX:SharedCacheHardLimit' in data:
        params.append('-XX:SharedCacheHardLimit=' + str(data['-XX:SharedCacheHardLimit']) + 'm')
        del data['-XX:SharedCacheHardLimit']

    if '-Xscmx' in data:
        params.append('-Xscmx=' + str(data['-Xscmx']) + 'm')
        del data['-Xscmx']

    if '-Dhttp.keepalive' in data:
        params.append('-Dhttp.keepalive=' + str(data['-Dhttp.keepalive']))
        del data['-Dhttp.keepalive']

    if '-Dhttp.maxConnections' in data:
        params.append('-Dhttp.maxConnections=' + str(data['-Dhttp.maxConnections']))
        del data['-Dhttp.maxConnections']

    if '-Xtune:virtualized' in data:
        if data['-Xtune:virtualized']:
            params.append('-Xtune:virtualized')
        del data['-Xtune:virtualized']

    if '-Xnojit' in data:
        if data['-Xnojit']:
            params.append('-Xnojit')
        del data['-Xnojit']

    if '-Xnoaot' in data:
        if data['-Xnoaot']:
            params.append('-Xnoaot')
        del data['-Xnoaot']

    return params + [item for item in data.values()]
