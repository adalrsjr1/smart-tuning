import unittest
from seqkmeans import Metric, Container, Cluster
import seqkmeans as skm
import pandas as pd
import config
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

    def test_metric_lt_number(self):
        m1 = Metric(1, 2, 3, 4)

        self.assertTrue(m1 < 5)

    def test_metric_gt_number(self):
        m1 = Metric(1, 2, 3, 4)
        m2 = Metric(1, 2, 3, 4)

        self.assertTrue(m1 < m2 * 3)

    def test_save_metric(self):
        db = config.client[config.MONGO_DB]
        collection = db.prod_metric_collection
        m1 = Metric(1, 2, 3, 4)
        m2 = Metric(5, 6, 7, 8)
        collection.insert_one({'time': 0, 'prod_metric': m1.__dict__, 'tuning_metric': m2.__dict__})

    def test_sum_metric_scalar(self):
        m1 = Metric(1,2,3,4)

        self.assertEqual(Metric(6,7,8,9), m1+5)

    def test_mult_metric_scalar(self):
        m1 = Metric(1,2,3,4)

        self.assertEqual(Metric(5,10,15,20), m1*5)

    def test_div_metric_metric(self):
        m1 = Metric(12, 12, 35, 40)
        m2 = Metric(5, 6, 7, 8)

        self.assertEqual(Metric(2.4, 2, 5, 5), m1 / m2)
        self.assertEqual(Metric(2, 2, 5, 5), m1 // m2)

    def test_cluster(self):
        cluster = Cluster()
        s1 = pd.Series(data=[1,2,5], index=['a', 'b','c'], name='s1', dtype=float)
        s2 = pd.Series(data=[1,3], index=['a', 'b'], name='s1', dtype=float)

        cluster.add(Container(label='s1', content=s1))
        cluster.add(Container(label='s2', content=s2))

        self.assertListEqual(pd.Series(data=[0.5,1.5,0.0], index=['a', 'b','c'], name='s1', dtype=float).tolist(),
                         cluster.centroid().to_list())

    def test_distance(self):
        s1 = pd.Series(data=[0, 0], index=['a', 'b'], name='s1', dtype=float)
        s2 = pd.Series(data=[3, 4], index=['a', 'b'], name='s1', dtype=float)

        print(s1.index.to_list(), s1.to_list())

        self.assertEqual(5.0, skm.__distance__(s1, s2, 'euclidean'))

if __name__ == '__main__':
    unittest.main()