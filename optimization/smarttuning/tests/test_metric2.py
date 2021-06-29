import unittest
from unittest import mock
from unittest.mock import MagicMock, PropertyMock

import config
from models.configuration import Configuration
from models.instance import Instance
from models.metric2 import Sampler, MetricDecorator, cfg_query


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config.init_k8s()

    def test_parse_json(self):
        instance = Instance(name='daytrader-service', namespace='default', is_production=True, sample_interval_in_secs=600,
                            ctx=None)
        sampler = Sampler(instance=instance, metric_schema_filepath=config.SAMPLER_CONFIG,
                          prom_url='http://localhost:30099')
        metric = sampler.sample()

    def test_evaluation(self):
        instance = Instance(name='daytrader-service', namespace='default', is_production=True, sample_interval_in_secs=100.2,
                            ctx=None)
        sampler = Sampler(instance, metric_schema_filepath=config.SAMPLER_CONFIG, prom_url='http://localhost:30099')
        metric = sampler.sample()
        metric_decorator = MetricDecorator(metric, sampler.objective_expr, sampler.penalization_expr)
        self.assertEqual(100, sampler.interval)
        print(metric_decorator.waiting_time)

    def test_cfg_sampling(self):
        instance = Instance(name='daytrader-service', namespace='default', is_production=True, sample_interval_in_secs=100.2,
                            ctx=None)

        instance.shutdown = MagicMock()
        with mock.patch('models.instance.Instance.configuration', new_callable=PropertyMock) as mock_configuration:
            mock_configuration.return_value = MagicMock()
            instance.configuration.data = {'daytrader-config-app': {'CONMGR1_MAX_POOL_SIZE': 33}}
            sampler = Sampler(instance, metric_schema_filepath=config.SAMPLER_CONFIG, prom_url='http://localhost:30099')
            self.assertDictEqual({'daytrader-config-app': {'CONMGR1_MAX_POOL_SIZE': 33}}, sampler.cfg())
            metric = sampler.sample()
            print('>>>', metric)
            self.assertEqual(33, metric.jdbc_connections)

if __name__ == '__main__':
    unittest.main()
