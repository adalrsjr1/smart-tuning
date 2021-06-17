# TODO: delete Saturation.py HPASampler.py Sampler.py
from __future__ import annotations

import json
import math
from collections import namedtuple

from kubernetes.client import AutoscalingV2beta2Api, V2beta2HorizontalPodAutoscaler, V2beta2MetricSpec, \
    V2beta2MetricStatus
from prometheus_pandas import query as handler

import config


def validate_json(j: dict) -> dict:
    expected = {'objective', 'saturation', 'metrics'}
    keys = set(j.keys())

    assert expected == keys, f"must have all these three keys: 'objective', 'saturation', 'metrics'"
    assert len(j['metrics']) > 0, f"must have at least one metric"

    return j


class Sampler:

    def __init__(self, podname: str, namespace: str, interval: int, metric_schema_filepath: str,
                 prom_url: str, training: bool = False):
        with open(metric_schema_filepath, 'r') as f:
            self.__raw = validate_json(json.load(f))
        self.podname = podname
        self.namespace = namespace
        self.interval = round(interval)
        self.training = training
        self.__prom_url = prom_url

    @property
    def objective_expr(self):
        return self.__raw['objective']

    @property
    def saturation_expr(self):
        return self.__raw['saturation']

    def prom(self) -> handler.Prometheus:
        return handler.Prometheus(self.__prom_url)

    def hpa(self) -> AutoscalingV2beta2Api:
        return config.hpaApi()

    def sample(self):
        metrics = {}
        for item in self.__raw['metrics']:

            key, value = query(self, **item)
            metrics[key] = value

        Metric2 = namedtuple('Metric2', list(metrics.keys()))

        return Metric2(**metrics)


class MetricDecorator:
    def __init__(self, ctx, objective_expr: str, saturation_expr: str):
        self.__ctx = {field: ctx[idx] for idx, field in enumerate(ctx._fields)}
        self.__objective_expr: str = objective_expr
        self.__saturation_expr: str = saturation_expr
        self.__dict__.update(self.__ctx)

    def objective(self) -> float:
        data = dict(self.__ctx)
        data.update({'saturation': self.saturation()})
        return eval(self.__objective_expr, globals(), data)

    def saturation(self) -> float:
        return eval(self.__saturation_expr, globals(), self.__ctx)

    def serialize(self) -> dict:
        data = dict(self.__ctx)
        data.update({'objective': self.objective(), 'saturation': self.saturation()})
        return data

def query(ctx: Sampler, name: str, query: str, datasource: str) -> (str, float):
    assert datasource in ['prom', 'hpa', 'scalar'], f'datasource must be either "prom" or "hpa" or "scalar", not "{datasource}"'
    if 'prom' == datasource:
        return prom_query(ctx, name, query)
    elif 'hpa' == datasource:
        return hpa_query(ctx, name, query)
    elif 'scalar' == datasource:
        return scalar_query(ctx, name, query)


def prom_query(ctx: Sampler, name: str, _query: str):
    result = ctx.prom().query(eval(_query, globals(), ctx.__dict__))
    if len(result) <= 0 or math.isnan(result): result = 0
    return name, float(result)


def hpa_query(ctx: Sampler, name: str, query: str):
    """
    queries:
      max_replicas
      current_replicas
      resource.current.<name>.average_utilization
      resource.target.<name>.average_utilization
    """

    def interpret_query(autoscaler: V2beta2HorizontalPodAutoscaler, query: str) -> float:
        def max_replicas(autoscaler: V2beta2HorizontalPodAutoscaler, query: str) -> float:
            return autoscaler.spec.max_replicas

        def current_replicas(autoscaler: V2beta2HorizontalPodAutoscaler, query: str) -> float:
            return autoscaler.status.current_replicas

        def resource_current(autoscaler: V2beta2HorizontalPodAutoscaler, query: str) -> float:
            resource_name = query.split('.')[2]
            metric: V2beta2MetricStatus
            for metric in autoscaler.status.current_metrics:
                if resource_name == metric.resource.name:
                    return metric.resource.current.average_utilization
            return float('nan')

        def resource_target(autoscaler: V2beta2HorizontalPodAutoscaler, query: str) -> float:
            resource_name = query.split('.')[2]
            metric: V2beta2MetricSpec
            for metric in autoscaler.spec.metrics:
                if resource_name == metric.resource.name:
                    return metric.resource.target.average_utilization
            return float('nan')

        if 'max_replicas' == query:
            return max_replicas(autoscaler, query)
        elif 'current_replicas' == query:
            return current_replicas(autoscaler, query)
        elif query.startswith('resource.current'):
            return resource_current(autoscaler, query)
        elif query.startswith('resource.target'):
            return resource_target(autoscaler, query)
        else:
            raise TypeError(f'can\'t process the query: {query}')

    raw: V2beta2HorizontalPodAutoscaler = ctx.hpa().read_namespaced_horizontal_pod_autoscaler(name=ctx.podname,
                                                                                              namespace=ctx.namespace)
    return name, interpret_query(raw, query)


def scalar_query(ctx: Sampler, name: str, query: str):
    return name, float(query)
