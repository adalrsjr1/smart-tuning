import unittest

import config
from models.metric2 import Sampler, MetricDecorator


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config.init_k8s()

    def test_parse_json(self):
        sampler = Sampler(podname='daytrader-service',
                          namespace='default',
                          interval=600,
                          metric_schema_filepath=config.SAMPLER_CONFIG,
                          prom_url='http://localhost:30099')
        metric = sampler.sample()

    def test_evaluation(self):
        sampler = Sampler(podname='daytrader-service',
                          namespace='default',
                          interval=100.2,
                          metric_schema_filepath=config.SAMPLER_CONFIG,
                          prom_url='http://localhost:30099')
        metric = sampler.sample()
        metric_decorator = MetricDecorator(metric, sampler.objective_expr, sampler.saturation_expr)
        print(metric_decorator.waiting_time)

if __name__ == '__main__':
    unittest.main()
