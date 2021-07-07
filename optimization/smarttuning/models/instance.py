from __future__ import annotations

import datetime
import heapq
import logging
import time

from kubernetes.client import ApiException, V2beta1HorizontalPodAutoscaler

import config
from typing import TYPE_CHECKING

from models.metric2 import Sampler, MetricDecorator

if TYPE_CHECKING:
    from controllers.searchspace import SearchSpaceContext
from models.configuration import Configuration, DefaultConfiguration
from sampler import PrometheusSampler, Metric

logger = logging.getLogger('models.smarttuning.ibm')
logger.setLevel('DEBUG')


class Instance:
    def __init__(self,
                 name: str,
                 namespace: str,
                 is_production: bool,
                 sample_interval_in_secs: int,
                 ctx: SearchSpaceContext,
                 sampler: PrometheusSampler = None):

        self._name = name
        self._namespace = namespace
        self._is_production = is_production
        self._ctx: SearchSpaceContext = ctx
        self._default_sample_interval = sample_interval_in_secs
        # self._sampler: PrometheusSampler = \
        #     PrometheusSampler(self.name, self._default_sample_interval) if sampler is None else sampler
        if sampler:
            self._sampler = sampler
        else:
            self._sampler = Sampler(instance=self, metric_schema_filepath=config.SAMPLER_CONFIG,
                                    prom_url=f'http://{config.PROMETHEUS_ADDR}:{config.PROMETHEUS_PORT}')
        self._active = True
        self._curr_config = None
        self._last_config = None
        self._config_counter = 0
        self._cache = {}

    def serialize(self) -> dict:
        serialized = {
            'name': self.name,
            'namespace': self.namespace,
            'production': self.is_production,
            'metric': self.metrics().serialize(),
            'curr_config': self.configuration.serialize() if self.configuration is not None else {},
            'last_config': self.last_config.serialize() if self.last_config is not None else {},
        }
        return serialized

    # def set_default_config(self, metrics: Metric):
    #     # eval option data
    #     current_config = self._ctx.get_current_config()
    #     strials = self._ctx.get_smarttuning_trials()
    #     self.configuration = strials.add_default_config(data=current_config, metric=metrics,
    #                                                     workload=self._ctx.workload)

    @property
    def sampler(self) -> Sampler:
        return self._sampler

    @property
    def name(self) -> str:
        return self._name

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def is_production(self) -> bool:
        return self._is_production

    @property
    def default_sample_interval(self) -> int:
        return round(self._default_sample_interval)

    @property
    def configuration(self) -> Configuration:
        return self._curr_config

    @property
    def config_counter(self):
        return self._config_counter

    @configuration.setter
    def configuration(self, new_config: Configuration):
        self.__configuration(new_config)
        self.patch_config(new_config)

    def __configuration(self, new_config: Configuration):
        self._config_counter += 1
        if self._curr_config is not new_config:
            self._config_counter = 0
        self._cache = {}
        self._last_config = self._curr_config
        self._curr_config = new_config

    def set_initial_configuration(self, configuration: Configuration, driver):
        self.__configuration(configuration)
        driver.session().study.add_trial(configuration.trial)
        # doens't save initial config in nursery to avoid ST discard reasonable configs because they are
        # slight worse than the intial config
        # heapq.heappush(driver.session().nursery, configuration)

    @property
    def last_config(self) -> Configuration:
        return self._last_config

    def patch_current_config(self):
        if self.configuration:
            self.patch_config(self.configuration)
        else:
            logger.warning(f'null configuration to update {self.name}')

    def patch_config(self, config_to_apply: Configuration):
        if self.active:
            try:
                manifests = self._ctx.model.manifests
                # check if is applying correct data if default config
                # default config keep indexed and values data
                data_to_apply = config_to_apply.data if not isinstance(config_to_apply,
                                                                       DefaultConfiguration) else config_to_apply.data
                self._do_patch(manifests, data_to_apply)
            except Exception:
                logger.exception(f'error when patching config:{config_to_apply.data} to {self.name}')

    def _do_patch(self, manifests, configuration):
        for key, value in configuration.items():
            for manifest in manifests:
                if key == manifest.name:
                    manifest.patch(value, production=self.is_production)
                    time.sleep(1)

    def metrics(self) -> MetricDecorator:
        # metric = self._sampler.
        return MetricDecorator(self._sampler.sample(), self._sampler.objective_expr, self._sampler.penalization_expr)

    # def workload(self, interval: int = 0) -> Future:
    #     if not self.active:
    #         return Future()
    #
    #     self._sampler.interval = self._default_sample_interval
    #     if interval > 0:
    #         self._sampler.interval = interval
    #
    #     return self._sampler.workload()

    @property
    def active(self) -> bool:
        return self._active

    def restart(self):
        logger.warning(f'restarting {self.name}')
        try:
            config.appsApi().patch_namespaced_deployment(name=self.name, namespace=self.namespace,
                                                         body={
                                                             "spec": {"template": {"metadata": {"annotations": {
                                                                 "kubectl.kubernetes.io/restartedAt":
                                                                     datetime.datetime.now(
                                                                         datetime.timezone.utc).isoformat()
                                                             }}, }}})
        except ApiException:
            logger.exception(f'cannot restart deployment:  {self.name}')

    @property
    def max_replicas(self) -> int:
        from controllers import injector
        name = self.name.strip(injector.training_suffix())
        replicas = 1
        try:
            hpa: V2beta1HorizontalPodAutoscaler = config.hpaApi().read_namespaced_horizontal_pod_autoscaler(
                name, self.namespace)
            replicas = hpa.spec.max_replicas
        except ApiException:
            logger.exception(f'error when retrieving number replicas of {name}.{self.namespace}')
        finally:
            return replicas

    @max_replicas.setter
    def max_replicas(self, value: int):
        from controllers import injector
        name = self.name.strip(injector.training_suffix())
        try:
            body = {
                'apiVersion': 'autoscaling/v2beta2',
                'kind': 'HorizontalPodAutoScaler',
                'spec': {'maxReplicas': value}
            }
            logger.debug(f'increase the max number of replicas of {name}.{self.namespace} to {value}')
            config.hpaApi().patch_namespaced_horizontal_pod_autoscaler(name, self.namespace, body=body)
        except ApiException:
            logger.exception(f'error when assigning a new number of replicas for {name}.{self.namespace}')

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self.active:
            logger.info(f'deleting {self.name}')
            self._active = False
            try:
                config.appsApi().delete_namespaced_deployment(self.name, self.namespace)
            except ApiException:
                logger.exception(f'fail to delete {self.name} instance on k8s cluster')
