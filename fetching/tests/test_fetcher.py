from common.dataaccess import MongoAccessLayer, PrometheusAccessLayer, PrometheusResponse

import numpy as np
import unittest


class TestFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = PrometheusAccessLayer('localhost', 9090)

    def test_query_data(self):
        start = 1580820340
        end = 1580825850
        step = 1

        data = self.fetcher.query('jmeter_threads{state="active"}', start, end, step)
        self.assertNotEqual(data.length(), 0)

    def test_query_empty(self):
        start = 0
        end = 1
        step = 1

        data = self.fetcher.query('', start, end, step)
        self.assertEqual(data.length(), 0)

    def test_group_subintervals(self):
        response = PrometheusResponse(np.arange(9))
        means = response.split_and_group(3, np.mean)

        for a, b in zip(means, [1, 4, 7]):
            self.assertEqual(a, b)

    def test_group_subintervals_size_not_multiple_size(self):
        response = PrometheusResponse(np.arange(8))
        means = response.split_and_group(3, np.mean)

        for a, b in zip(means, [1.5, 5.5]):
            self.assertEqual(a, b)

    def test_group(self):
        response = PrometheusResponse(np.arange(8))
        self.assertEqual(response.group(np.sum), 28)
