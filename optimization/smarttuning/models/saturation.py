from sampler import Metric


class Saturation:
    def __init__(self, name: str, expression: str, target: float):
        self.__name = name
        self.__expression = expression
        self.__target = target

    @property
    def name(self):
        return self.__name

    @property
    def target(self) -> float:
        return self.__target

    def current(self) -> float:
        prom_metric_dict = prom_metric.to_dict()
        hpa_metric_dict = hpa_metric.to_dict()
        local_ctx = {}
        local_ctx.update(prom_metric_dict)
        local_ctx.update(hpa_metric_dict)

        return eval(self.__expression, __globals=globals(), __locals=local_ctx)

    def is_saturated(self, prom_metric: Metric, hpa_metric: HpaMetric) -> bool:
        return self.target >= self.current
