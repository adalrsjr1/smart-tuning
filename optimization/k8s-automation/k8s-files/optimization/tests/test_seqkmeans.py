import unittest
from seqkmeans import Metric, Container
from prometheus_pandas import query
import sampler

class TestSeqKmeans(unittest.TestCase):

    def test_metric(self):
       metric = Metric()
       self.assertIsNot(metric, None)

    def test_sum_metric_metric(self):
        m1 = Metric(1,2,3,4)
        m2 = Metric(5,6,7,8)

        self.assertEqual(Metric(6,8,10,12), m1+m2)

    def test_sum_metric_scalar(self):
        m1 = Metric(1,2,3,4)

        self.assertEqual(Metric(6,7,8,9), m1+5)

    def test_div_metric_metric(self):
        m1 = Metric(12, 12, 35, 40)
        m2 = Metric(5, 6, 7, 8)

        self.assertEqual(Metric(2.4, 2, 5, 5), m1 / m2)
        self.assertEqual(Metric(2, 2, 5, 5), m1 // m2)

    def test_create_container(self):
        prometheus = query.Prometheus(f'http://localhost:30090')
        hist = sampler.workload('acmeair-tuning.*', 600)
        print(hist.result())
        c = Container('', hist)

if __name__ == '__main__':
    unittest.main()
