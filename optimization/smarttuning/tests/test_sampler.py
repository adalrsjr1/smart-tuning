import unittest
from sampler import PrometheusSampler, Metric
import pandas as pd
import sampler
import math
import json

class TestPrometheusSampling(unittest.TestCase):

    def client(self):
        return PrometheusSampler('acmeair-tuning-.*', interval=900, addr='localhost', port='30099')

    def test_sample_parsing(self):
        instr = '{dst="10.244.4.17",instance="10.244.4.17:9090",path="/customer/byid/uid45@email.com=300",pod="acmeair-customerservicesmarttuning-dd64b486-8qbmm",service="acmeair-customer-servicesmarttuning",src="10.244.2.9"}'
        output = '{"dst":"10.244.4.17","instance":"10.244.4.17:9090","path":"/customer/byid/uid45@email.com=300","pod":"acmeair-customerservicesmarttuning-dd64b486-8qbmm","service":"acmeair-customer-servicesmarttuning","src":"10.244.2.9"}'

        self.assertDictEqual(json.loads(output), sampler.parser_to_dict(instr))

    def test_prom_sample(self):
        prom = self.client()
        self.assertIsInstance(prom, PrometheusSampler)

    def test_sample_throuhgput(self):
        prom = self.client()
        future = prom.throughput()
        result = sampler.__extract_value_from_future__(future, 1000)
        self.assertGreaterEqual(result, 0)

    def test_sample_error(self):
        prom = self.client()
        future = prom.error()
        result = sampler.__extract_value_from_future__(future, 1000)
        self.assertGreaterEqual(result, 0)

    def test_sample_latency(self):
        prom = self.client()
        future = prom.process_time()
        result = sampler.__extract_value_from_future__(future, 1000)
        self.assertGreaterEqual(result, 0)

    def test_sample_memory(self):
        prom = self.client()
        future = prom.memory()
        result = sampler.__extract_value_from_future__(future, 1000)
        self.assertGreaterEqual(result, 0)

    def test_sample_cpu(self):
        prom = self.client()
        future = prom.cpu()
        result = sampler.__extract_value_from_future__(future, 1000)
        self.assertGreaterEqual(result, 0)

    def test_sample_workload(self):
        prom = self.client()
        future = prom.workload()
        result = future.result()
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(result)
        result = sampler.series_to_dataframe(result)
        self.assertGreaterEqual(len(result), 0)

    def test_sample_empty_workload(self):
        import time
        prom = self.client()
        future = prom.workload()
        result = future.result()
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(result)
        result = sampler.series_to_dict(result)
        self.assertEqual(len(result), 0)

    def test_sample_metric(self):
        prom = self.client()
        metric = prom.metric()
        self.assertGreaterEqual(metric.cpu(), 0)
        self.assertGreaterEqual(metric.memory(), 0)
        self.assertGreaterEqual(metric.process_time(), 0)
        self.assertGreaterEqual(metric.throughput(), 0)
        self.assertGreaterEqual(metric.errors(), 0)

        print(metric)

    def test_metric_operation(self):
        m1 = Metric(cpu=1, memory=1, throughput=1, process_time=1, in_out=1, errors=1)
        m2 = Metric(cpu=3, memory=3, throughput=3, process_time=3, in_out=3, errors=3)

        result = m1.__operation__(m2, lambda a, b: a + b)
        self.assertEqual(result, Metric(cpu=4, memory=4, throughput=4, process_time=4, in_out=4, errors=4))
        result = m1 + m2
        self.assertEqual(result, Metric(cpu=4, memory=4, throughput=4, process_time=4, in_out=4, errors=4))

        m1 = Metric(cpu=1, memory=1, throughput=1, process_time=1, errors=1)
        m2 = Metric(cpu=3, memory=3, throughput=3, process_time=3, errors=3)

        result = m1.__operation__(m2, lambda a, b: a + b)
        self.assertEqual(result, Metric(cpu=4, memory=4, throughput=4, process_time=4, errors=4))
        result = m1 + m2
        self.assertEqual(result, Metric(cpu=4, memory=4, throughput=4, process_time=4, errors=4))


    def test_metric_operation_scalar(self):
        m1 = Metric(cpu=1, memory=1, throughput=1, process_time=1, in_out=1, errors=1)
        result = m1 * 4
        self.assertEqual(result, Metric(cpu=4, memory=4, throughput=4, process_time=4, errors=4, in_out=4))

        m1 = Metric(cpu=1, memory=1, throughput=1, process_time=1, errors=1)
        result = m1 * 4
        self.assertEqual(result, Metric(cpu=4, memory=4, throughput=4, process_time=4, errors=4))

    def test_metric_logic_op(self):
        m1 = Metric(cpu=1, memory=1, throughput=1, process_time=1, in_out=1, errors=1,
                    to_eval='cpu+memory+throughput+process_time+in_out+errors')
        m2 = Metric(cpu=3, memory=3, throughput=3, process_time=3, errors=3,
                    to_eval='cpu+memory+throughput+process_time+errors')

        m3 = Metric(cpu=3, memory=3, throughput=3, process_time=3, errors=3,
                    to_eval='cpu+memory+throughput+process_time+in_out+errors')

        print(m1)
        print(m2)
        self.assertEqual(6, m1.objective())
        self.assertLess(m1, m2)
        self.assertLessEqual(m1, m2)
        self.assertLessEqual(m1, m1)
        self.assertGreater(m2, m1)
        self.assertGreaterEqual(m2, m2)
        self.assertTrue(math.isinf(m3.objective()))

    def test_objective(self):
        m = Metric(cpu=1, memory=1, throughput=1, process_time=1, in_out=0, errors=1,
                   to_eval='cpu/in_out')
        self.assertTrue(math.isinf(m.objective()))

        m = Metric(cpu=1, memory=1, throughput=1, process_time=1, in_out=0, errors=1,
                   to_eval='cpu / (10e-15 if in_out == 0 else in_out)')
        print(m.objective())
        self.assertFalse(math.isinf(m.objective()))

    def test_metric_serialization(self):
        print(self.client().metric().serialize())


if __name__ == '__main__':
    unittest.main()
