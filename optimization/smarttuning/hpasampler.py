from collections import namedtuple
from typing import Union

from kubernetes.client import V2beta2HorizontalPodAutoscaler, V2beta2HorizontalPodAutoscalerSpec, \
    V2beta2HorizontalPodAutoscalerStatus, V2beta2MetricSpec, V2beta2ResourceMetricStatus, V2beta2MetricValueStatus

import config


class HpaSampler:
    def __init__(self, name: str, namespace: str):
        self.__name = name
        self.__namespace = namespace
        self.__raw: V2beta2HorizontalPodAutoscaler = api.read_namespaced_horizontal_pod_autoscaler(
            name=name,
            namespace=namespace)

    @property
    def name(self):
        return self.__name

    @property
    def namespace(self):
        return self.__namespace

    @property
    def spec(self) -> V2beta2HorizontalPodAutoscalerSpec:
        return self.__raw.spec

    @property
    def status(self) -> V2beta2HorizontalPodAutoscalerStatus:
        return self.__raw.status

    def resources(self, to_dict: bool = False) -> Union[dict[str, tuple[str, float, float, float]],
                                                        list[tuple[str, float, float, float]]]:
        HpaResource = namedtuple('HpaResource', ['name', 'current', 'target', 'value'])
        _resources = []
        for _spec, _status in zip(self.spec.metrics, self.status.current_metrics):
            status_resource = _status.resource or V2beta2ResourceMetricStatus(
                current=V2beta2MetricValueStatus(0,0,0),
                name=_spec.resource.name)
            current = status_resource.current.average_utilization
            spec_resource: V2beta2MetricSpec = _spec.resource
            target = spec_resource.target.average_utilization
            resource = HpaResource(
                name=spec_resource.name,
                current=current,
                target=target,
                value=current - target)
            _resources.append(resource)
        _resources.append(HpaResource(name='replicas',
                                      target=self.spec.max_replicas,
                                      current=self.status.current_replicas,
                                      value=self.status.current_replicas-self.spec.max_replicas))

        if to_dict:
            return {r.name: r for r in _resources}
        return _resources


if __name__ == '__main__':
    from pprint import pprint
    config.init_k8s()
    api = config.hpaApi()
    name = 'nginx'
    # hpa: V2beta2HorizontalPodAutoscaler = api.read_namespaced_horizontal_pod_autoscaler(name=name, namespace='default')
    # spec: V2beta2HorizontalPodAutoscalerSpec = hpa.spec
    # status: V2beta2HorizontalPodAutoscalerStatus = hpa.status

    sampler = HpaSampler(name=name, namespace='default')
    pprint(sampler.resources(to_dict=True))

    # from pprint import pprint
    #
    # pprint(hpa)
    #
    # print('--- status ---')
    # for metric in status.current_metrics:
    #     pprint(metric)
    #
    # print('--- spec --- ')
    # for metric in spec.metrics:
    #     pprint(metric)
