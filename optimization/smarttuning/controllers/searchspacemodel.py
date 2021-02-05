from __future__ import annotations

import abc
import copy
import json
import logging
import random
import time
from distutils.util import strtobool
from typing import Union
from unittest import TestCase

import optuna
from kubernetes.client.models import *
from kubernetes.utils import quantity
from optuna.distributions import UniformDistribution, DiscreteUniformDistribution, IntUniformDistribution, \
    CategoricalDistribution

import config

logger = logging.getLogger(config.SEARCH_SPACE_LOGGER)
logger.setLevel(logging.DEBUG)


class SearchSpaceModel:
    def __init__(self, o, study: optuna.Study):
        self.study = study
        if o:
            temp = parse_manifests(o, self)

            self.deployment: str = temp['deployment']
            self.manifests: list[BaseSearchSpaceModel] = temp['models']
            self.namespace: str = temp['namespace']
            self.service: str = temp['service']

    def adhoc_trial(self) -> optuna.trial.FixedTrial:
        memo = {}
        params = {}

        for name, tunable in self.tunables().items():
            params.update({name: tunable.sample(trial=None, memo=memo)})

        return optuna.trial.FixedTrial(params)

    def default_structure(self, plain_config:dict) -> dict:
        tunables = self.tunables()
        _sample = {}
        for manifest in self.manifests:
            name = manifest.name
            ss = {name: {}}
            tunables = manifest.tunables
            for key, tunable in tunables.items():
                ss[name][key] = plain_config[key]
            _sample.update(ss)

        return _sample

    def tunables(self) -> dict[str, BaseRangeModel]:
        manifest: BaseSearchSpaceModel
        all_tunables = {}
        for manifest in self.manifests:
            all_tunables.update(manifest.tunables)

        return all_tunables

    def sample(self, trial: optuna.trial.BaseTrial, full=False):
        """
        trial: optuna.trial.BaseTrial
        full: if true return the tunables hierarchy {'manifest-name': {'tunable-name': value}}
        otherwise return a plain list of tunables {'tunable-name': 'value'}
        """
        tunable: BaseRangeModel
        _sample = {}
        memo = {}
        if full:
            for manifest in self.manifests:
                name = manifest.name
                ss = {name: {}}
                tunables = manifest.tunables
                for key, tunable in tunables.items():
                    try:
                        ss[name][key] = tunable.sample(trial, memo)
                    except:
                        logger.exception(f'{name}:{key}')
                        exit(1)
                _sample.update(ss)
        else:
            for key, tunable in self.tunables().items():
                _sample[key] = tunable.sample(trial, memo)

        return _sample


def parse_manifests(o: dict, ctx: SearchSpaceModel) -> dict:
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
                    models.append(DeploymentSearchSpaceModel(name, namespace, deployment, tunables, ctx))
                elif 'configMap' == manifest_type:
                    models.append(ConfigMapSearhSpaceModel(name, namespace, deployment, filename, tunables, ctx))

    return {'namespace': namespace, 'service': service, 'deployment': context, 'models': models}


class BaseSearchSpaceModel:
    def __init__(self, name: str, namespace: str, tunables: dict[str, BaseRangeModel], ctx: SearchSpaceModel):
        self.name: str = name
        self.ctx: SearchSpaceModel = ctx
        self.namespace: str = namespace
        self.tunables: dict[str, BaseRangeModel] = parse_tunables(tunables, self.ctx)

    def get_current_config(self):
        pass

    def sample(self):
        pass


def parse_tunables(tunables, ctx: SearchSpaceModel) -> dict[str, BaseRangeModel]:
    parsed_tunables: dict[str, BaseRangeModel] = {}
    for type_range, list_tunables in tunables.items():
        for r in list_tunables:
            if 'boolean' == type_range:
                r['values'] = ['True', 'False']
                r['type'] = 'bool'
                parsed_tunables[r['name']] = OptionRangeModel(r, ctx)
            elif 'number' == type_range:
                parsed_tunables[r['name']] = NumberRangeModel(r, ctx)
            elif 'option' == type_range:
                parsed_tunables[r['name']] = OptionRangeModel(r, ctx)

    return parsed_tunables


class DeploymentSearchSpaceModel(BaseSearchSpaceModel):
    def __init__(self, name: str, namespace: str, deployment: str, tunables: dict[str, BaseRangeModel],
                 ctx: SearchSpaceModel):
        super(DeploymentSearchSpaceModel, self).__init__(name=name, namespace=namespace, tunables=tunables, ctx=ctx)
        self.deployment = deployment

    def __repr__(self):
        return f'{{"name": "{self.name}", "namespace": "{self.namespace}", "deployment": "{self.deployment}"' \
               f'tunables: {str(self.tunables)}}}'

    def get_current_config(self) -> (dict, dict):
        api_instance = config.appsApi()
        deployment: V1Deployment = api_instance.read_namespaced_deployment(name=self.name, namespace=self.namespace)

        return self.get_current_config_from_spec(deployment)

    def get_current_config_from_spec(self, deployment: V1Deployment) -> dict:
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

        return {k: config_limits[k] for k, v in self.tunables.items() if k in config_limits}

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


class ConfigMapSearhSpaceModel(BaseSearchSpaceModel):
    def __init__(self, name: str, namespace: str, deployment: str, filename: str, tunables: dict[str, BaseRangeModel],
                 ctx: SearchSpaceModel):
        super(ConfigMapSearhSpaceModel, self).__init__(name=name, namespace=namespace, tunables=tunables, ctx=ctx)
        self.deployment = deployment
        self.filename = filename

    def __repr__(self):
        return f'{{"name": "{self.name}", "namespace": "{self.namespace}", "deployment": "{self.deployment}"' \
               f'tunables: {str(self.tunables)}}}'

    def get_current_config(self) -> dict:
        core_api = config.coreApi()
        config_map: V1ConfigMap = core_api.read_namespaced_config_map(name=self.name, namespace=self.namespace)

        return self.get_current_config_core(config_map)

    def get_current_config_core(self, config_map: V1ConfigMap) -> dict:
        # if Option set index to default config
        # otherwise set values
        if 'jvm.options' in config_map.data:
            # TODO: workaround to get inital config from jvm
            if len(config_map.data['jvm.options']) == 0:
                raise RuntimeError(f'config file: {config_map.data} cannot be empty -- set default config')
            else:
                keys, config_map.data = jmvoptions_to_dict(config_map.data['jvm.options'])

        return {k: config_map.data[k] for k, v in self.tunables.items() if k in config_map.data}

    def patch(self, new_config: dict, production=False):
        if self.filename == 'jvm.options':
            return self.__patch_jvm(new_config, production)
        elif self.filename == '':
            return self.__patch_envvar(new_config, production)

    def __patch_envvar(self, new_config, production=False):
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

    def __core_patch_jvm(self, new_config, production=False):
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

    def __patch_jvm(self, new_config, production=False):
        name, namespace, body = self.__core_patch_jvm(new_config, production)

        return config.coreApi().patch_namespaced_config_map(name, namespace, body, pretty='false')


class BaseRangeModel:
    def __init__(self, name, ctx: SearchSpaceModel):
        self.name = name
        self.ctx = ctx

    @abc.abstractmethod
    def sample(self, trial: optuna.trial.BaseTrial, memo:dict) -> Union[int, float, str]:
        return

    @abc.abstractmethod
    def distribution(self, trial: optuna.trial.BaseTrial, memo: dict) -> optuna.distributions.BaseDistribution:
        return


class NumberRangeModel(BaseRangeModel):
    def __init__(self, r, ctx: SearchSpaceModel = None):
        super(NumberRangeModel, self).__init__(r['name'], ctx)
        self.upper, self.upper_dep = self.__unpack_value(r['upper'])
        self.lower, self.lower_dep = self.__unpack_value(r['lower'])
        self.step = r.get('step', None)
        self.real = r.get('real', False)

    def __unpack_value(self, item):
        if item and isinstance(item, dict):
            return item['value'], item['dependsOn']
        raise ValueError(
            f'{self.name} has a invalid item:{item}, expecting a dict:{{"dependsOn":"str", "value":"int-or-str"}}')

    def __repr__(self):
        return f'{{"name": "{self.name}", ' \
               f'"lower": {self.lower}, ' \
               f'"lower_dep":{self.lower_dep}, ' \
               f'"upper": {self.upper}, ' \
               f'"upper_dep":{self.upper_dep}, ' \
               f'"real": "{self.real}", ' \
               f'"step": {self.step}}}'

    def sample(self, trial: optuna.trial.BaseTrial, memo: dict) -> Union[int, float, str]:

        if self.name in memo:
            return memo[self.name]


        step = self.get_step()
        if isinstance(trial, optuna.trial.FixedTrial):
            d = self.distribution(None, memo)
            lower = d.low
            upper = d.high
        else:
            lower = self.get_lower(trial, memo, eval_dep=trial is not None)
            upper = self.get_upper(trial, memo, eval_dep=trial is not None)

            if lower > upper:
                step = None
                lower = upper

            assert lower <= upper, f' {self.name}: lower[{lower}] >= upper[{upper}]'
            # print('[s]', self.name, lower, upper, step)



        if trial:
            if self.get_real():
                value = trial.suggest_float(name=self.name,
                                            low=lower,
                                            high=upper,
                                            step=step,
                                            log=False)
            else:
                value = trial.suggest_int(name=self.name,
                                          low=lower,
                                          high=upper,
                                          step=step if step and step > 0 else 1,
                                          log=False)
        else:
            d = self.distribution(None, memo)
            low = d.low
            high = d.high
            q = 1.0
            if isinstance(d, DiscreteUniformDistribution):
                q = d.q
            elif isinstance(d, IntUniformDistribution):
                q = d.step

            if self.get_real():
                value = random.uniform(low, high) * q // q
            else:
                value = random.randint(low, high) * q // q

        memo[self.name] = value
        return value

    def distribution(self, trial: optuna.trial.BaseTrial, memo: dict) -> optuna.distributions.BaseDistribution:
        lower = self.get_lower(trial, memo, eval_dep=trial is not None)
        upper = self.get_upper(trial, memo, eval_dep=trial is not None)
        # print('[d]', self.name, lower, upper, self.get_step())
        step = self.get_step()
        if lower > upper:
            step = None
            lower = upper

        assert lower <= upper, f' {self.name}: lower[{lower}] >= upper[{upper}]'

        if self.get_real():
            if self.get_step() is None:
                return UniformDistribution(low=lower, high=upper)
            else:
                return DiscreteUniformDistribution(low=lower, high=upper, q=step)
        else:
            if self.get_step() is None or self.get_step() <= 1:
                return IntUniformDistribution(low=lower, high=upper)
            else:
                return IntUniformDistribution(low=lower, high=upper, step=step)

    def __is_a_numeric_string(self, value):
        return isinstance(value, str) and str.isalpha(value)

    def get_upper(self, trial: optuna.trial.BaseTrial, memo: dict, eval_dep=True):
        if self.name in memo:
            return memo[self.name]

        if not eval_dep or self.get_upper_dep() == '':
            if self.__is_a_numeric_string(self.upper):
                return float(self.upper) if self.get_real() else int(self.upper)
            return float(self.upper) if self.get_real() else int(self.upper)
        else:
            upper = dep_eval(self.get_upper_dep(), self.ctx, trial, memo)
            return float(upper) if self.get_real() else int(upper)

    def get_upper_dep(self):
        return self.upper_dep

    def get_lower(self, trial: optuna.trial.BaseTrial, memo: dict, eval_dep=True):
        if self.name in memo:
            return memo[self.name]

        if not eval_dep or self.get_lower_dep() == '':
            if self.__is_a_numeric_string(self.lower):
                return float(self.lower) if self.get_real() else int(self.lower)
            return float(self.lower) if self.get_real() else int(self.lower)
        else:
            lower = dep_eval(self.get_lower_dep(), self.ctx, trial, memo)
            return float(lower) if self.get_real() else int(lower)

    def get_lower_dep(self):
        return self.lower_dep

    def get_step(self):
        if self.step is None:
            return self.step
        return float(self.step)

    def get_real(self):
        if isinstance(self.real, bool):
            return self.real
        return strtobool(self.real)

def dep_eval(expr, ctx: SearchSpaceModel, trial: optuna.trial.BaseTrial, memo: dict):
    expr_lst = tokenize(expr, ctx, trial, memo)
    return polish_eval(expr_lst)


def tokenize(expr, ctx: SearchSpaceModel, trial: optuna.trial.BaseTrial, memo: dict):
    if len(expr) != 0:
        tokens = expr.split()
        return [eval_token(token, ctx, trial, memo) for token in tokens]
    return [eval_token('', ctx, trial, memo)]


def eval_token(token, ctx: SearchSpaceModel, trial: optuna.trial.BaseTrial, memo: dict):
    import re

    if token in memo:
        return memo[token]

    if re.compile(r"[-+]?\d*\.\d+|\d+").match(token):
        return float(token)
    else:
        if token in ['+', '-', '*', '/']:
            return token
        else:
            tunables = ctx.tunables()
            if token in tunables:
                return tunables[token].sample(trial, memo)

    raise TypeError(f'cant parse: "{token}" not defined as a tunable')


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


class OptionRangeModel(BaseRangeModel):
    def __init__(self, r, ctx: SearchSpaceModel = None):
        super(OptionRangeModel, self).__init__(r['name'], ctx)
        self.type = r['type']
        self.values = r['values']

    def __repr__(self):
        return f'{{"name": "{self.name}, "type":{self.type}, "values": {json.dumps(self.values)} }}'

    def sample(self, trial: optuna.trial.BaseTrial, memo: dict) -> Union[int, float, str]:
        if self.name in memo:
            return memo[self.name]
        if trial:
            value = trial.suggest_categorical(name=self.name, choices=self.get_values())
        else:
            value = random.choice(self.get_values())
        if 'integer' == self.type:
            value = int(value)
        elif 'real' == self.type:
            value = float(value)
        elif 'bool' == self.type:
            value = True if value == 1 or value == True else False
        else:
            value = str(value)

        memo[self.name] = value
        return value

    def distribution(self, trial: optuna.trial.BaseTrial, memo: dict) -> optuna.distributions.BaseDistribution:
        return CategoricalDistribution(self.get_values())

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
        types = {'integer': int, 'real': float, 'string': str, 'bool': strtobool}
        to_return = [types[new_type](value) for value in values]
        return to_return

    def get_values(self):
        return self.cast(self.values, self.type)


def jmvoptions_to_dict(jvm_options):
    data = jvm_options.split('\n')
    data = [item for item in data if not item.startswith('#')]
    params = {}
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
