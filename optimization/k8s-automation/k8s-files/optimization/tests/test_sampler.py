import unittest

from prometheus_pandas import query
import sampler

class TestPrometheusSampling(unittest.TestCase):
    prometheus = query.Prometheus(f'http://localhost:30090')

    def test_connection(self):
        future_sample = sampler.do_sample('up', endpoint=TestPrometheusSampling.prometheus)
        future_sample.result()

        self.assertGreater(future_sample.result().size, 0)
        
    def test_throughput(self):
        future_sample = sampler.throughput("acmeair-.*", 600)
        print(future_sample.result())

    def test_latency(self):
        future_sample = sampler.latency("acmeair-.*", 600)
        print(type(future_sample.result()))

    def test_memory(self):
        future_sample = sampler.memory("acmeair-db.*", 600)
        print(type(future_sample.result()))

    def test_cpu(self):
        future_sample = sampler.cpu("acmeair-tuning.*", 600)
        print(future_sample.result())

    def test_hist(self):
        future_sample = sampler.workload("acmeair-tuning.*", 900)
        print(future_sample.result())

if __name__ == '__main__':
    unittest.main()
