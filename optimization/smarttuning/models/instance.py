from __future__ import annotations

import datetime
import logging
import time
from concurrent.futures import Future

from kubernetes.client import ApiException

import config
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
        self._sampler: PrometheusSampler = PrometheusSampler(self.name,
                                                             self._default_sample_interval) if sampler is None else sampler
        self._active = True
        self._curr_config = None
        self._last_config = None
        self._config_counter = 0
        self._cache = {}

    def serialize(self) -> dict:
        return {
            'name': self.name,
            'namespace': self.namespace,
            'production': self.is_production,
            'metric': self.metrics(cached=True).serialize(),
            'curr_config': self.configuration.serialize() if not self.configuration is None else {},
            'last_config': self.last_config.serialize() if not self.last_config is None else {},
        }

    def set_default_config(self, metrics: Metric):
        # eval option data
        current_config = self._ctx.get_current_config()
        strials = self._ctx.get_smarttuning_trials()
        self.configuration = strials.add_default_config(data=current_config, metric=metrics)

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
        return self._default_sample_interval

    @property
    def configuration(self) -> Configuration:
        return self._curr_config

    @property
    def config_counter(self):
        return self._config_counter

    @configuration.setter
    def configuration(self, new_config: Configuration):
        self._config_counter += 1
        if self._curr_config is not new_config:
            self._config_counter = 0
        self._cache = {}
        self._last_config = self._curr_config
        self._curr_config = new_config
        self.patch_config(new_config)

    def update_configuration_score(self, metric: Metric):
        self.configuration.update_score(metric)

    @property
    def last_config(self) -> Configuration:
        return self._last_config

    def patch_config(self, config_to_apply: Configuration):
        if self.active:
            try:
                manifests = self._ctx.model.manifests
                # check if is applying correct data if default config
                # default config keep indexed and values data
                data_to_apply = config_to_apply.data if not isinstance(config_to_apply,
                                                                       DefaultConfiguration) else config_to_apply.data
                self._do_patch(manifests, data_to_apply)
            except:
                logger.exception(f'error when patching config:{config_to_apply.data}')

    def _do_patch(self, manifests, configuration):
        for key, value in configuration.items():
            for manifest in manifests:
                if key == manifest.name:
                    manifest.patch(value, production=self.is_production)
                    time.sleep(1)

    def metrics(self, interval: int = 0, cached=False) -> Metric:
        if cached and ('metrics', interval) in self._cache:
            metric = self._cache[('metrics', interval)]
            metric.set_restarts(self.configuration.n_restarts)
            return metric
        if not self.active:
            Metric.zero()

        self._sampler.interval = self._default_sample_interval
        if interval > 0:
            self._sampler.interval = interval

        self._cache[('metrics', interval)] = self._sampler.metric()
        if cached:
            metric = self._cache[('metrics', interval)]
            metric.set_restarts(self.configuration.n_restarts)
            return metric

        metric = self._sampler.metric()
        metric.set_restarts(self.configuration.n_restarts if self.configuration else 0)
        return metric

    def workload(self, interval: int = 0) -> Future:
        if not self.active:
            return Future()

        self._sampler.interval = self._default_sample_interval
        if interval > 0:
            self._sampler.interval = interval

        return self._sampler.workload()

    @property
    def active(self) -> bool:
        return self._active

    def restart(self):
        logger.warning(f'restarting {self.name}')
        self.configuration.increment_restart_counter()
        try:
            config.appsApi().patch_namespaced_deployment(name=self.name, namespace=self.namespace,
                                                         body={
                                                             "spec": {
                                                                 "template": {
                                                                     "metadata": {
                                                                         "annotations": {
                                                                             "kubectl.kubernetes.io/restartedAt": datetime.datetime.now(
                                                                                 datetime.timezone.utc).isoformat()
                                                                         }
                                                                     },
                                                                 }
                                                             }
                                                         })
        except ApiException:
            logger.exception(f'cannot restart deployment:  {self.name}')

    def shutdown(self):
        if not self.active:
            logger.info(f'deleting training:{self.name}')
            self._active = False
            config.appsApi().delete_namespaced_deployment(self.name, self.namespace)

    def __del__(self):
        self.shutdown()
