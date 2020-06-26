import unittest
from seqkmeans import Container, Cluster, __merge_data__
import seqkmeans as skm
import pandas as pd


class TestSeqKmeans(unittest.TestCase):

    def test_merging_data(self):
        s1 = pd.Series(data=[1, 2, 5], index=['a', 'b', 'c'], name='s1', dtype=float)
        s2 = pd.Series(data=[1, 3], index=['a', 'b'], name='s1', dtype=float)

        col0, col1, idx = __merge_data__(s1, s2)
        self.assertListEqual(col0.index.to_list(), col1.index.to_list())
        self.assertListEqual([value for i, value in col0.items()], [1, 2, 5])
        self.assertListEqual([value for i, value in col1.items()], [1, 3, 0])

        s1, s2 = s2, s1

        col0, col1, idx = __merge_data__(s1, s2)
        self.assertListEqual(col0.index.to_list(), col1.index.to_list())
        self.assertListEqual([value for i, value in col0.items()], [1, 3, 0])
        self.assertListEqual([value for i, value in col1.items()], [1, 2, 5])

        s2 = s1

        col0, col1, idx = __merge_data__(s1, s2)
        self.assertListEqual(col0.index.to_list(), col1.index.to_list())
        self.assertListEqual([value for i, value in col0.items()], [value for i, value in col1.items()])

    def test_merging_data0(self):
        s1 = pd.Series(data=[1, 2, 5], index=['a', 'b', 'c'], name='s1', dtype=float)
        s2 = pd.Series(data=[], index=[], name='s1', dtype=float)

        col0, col1, idx = __merge_data__(s1, s2)
        self.assertListEqual(col0.index.to_list(), col1.index.to_list())
        self.assertListEqual([value for i, value in col0.items()], [1, 2, 5])
        self.assertListEqual([value for i, value in col1.items()], [0, 0, 0])

        s1, s2 = s2, s1

        col0, col1, idx = __merge_data__(s1, s2)
        self.assertListEqual(col0.index.to_list(), col1.index.to_list())
        self.assertListEqual([value for i, value in col0.items()], [0, 0, 0])
        self.assertListEqual([value for i, value in col1.items()], [1, 2, 5])

        s2 = s1

        col0, col1, idx = __merge_data__(s1, s2)
        self.assertListEqual(col0.index.to_list(), col1.index.to_list())
        self.assertListEqual([value for i, value in col0.items()], [value for i, value in col1.items()])

    def load_requests(self):
        with open('requests.histogram') as f:
            index = []
            data = []
            for row in f:
                splited = row.split()
                index.append(splited[0])
                data.append(splited[1])

        return pd.Series(data=data, index=index, name='histogram', dtype=float)

    def test_grouping_rows(self):
        from pprint import pprint
        table = skm.__compare__(histograms=self.load_requests(), threshold=0.1)
        pprint(skm.__hist_reduce__(table))

    def test_cluster(self):
        cluster = skm.Cluster()
        s1 = pd.Series(data=[1, 2, 5], index=['a', 'b', 'c'], name='s1', dtype=float)
        s2 = pd.Series(data=[1, 3], index=['a', 'b'], name='s1', dtype=float)

        cluster.add(skm.Container(label='s1', content=s1))
        cluster.add(skm.Container(label='s2', content=s2))

        self.assertListEqual(pd.Series(data=[0.5, 1.5, 0.0], index=['a', 'b', 'c'], name='s1', dtype=float).tolist(),
                             cluster.centroid().to_list())

    def test_distance(self):
        s1 = pd.Series(data=[0, 0], index=['a', 'b'], name='s1', dtype=float)
        s2 = pd.Series(data=[3, 4], index=['a', 'b'], name='s1', dtype=float)

        self.assertEqual(5.0, skm.__distance__(s1, s2, 'euclidean'))


if __name__ == '__main__':
    unittest.main()
