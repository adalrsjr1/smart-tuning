import math
import numbers
import random
import numpy as np
import unittest
from unittest import TestCase

from util.stats import RunningStats


class TestRunningStats(TestCase):
    def test_accumulative_diff(self):
        rs = RunningStats()
        rs.push(1987698.987)
        self.assertEqual(0, rs.accumulative_diff())

        rs.push(1987870.987)
        self.assertEqual(172, rs.accumulative_diff())

        rs.push(1987990.987)
        self.assertEqual(292, rs.accumulative_diff())

        rs.push(1988200.987)
        self.assertEqual(502, rs.accumulative_diff())

        rs.push(1988000.987)
        self.assertEqual(302, rs.accumulative_diff())

        rs = RunningStats()
        [rs.push(n) for n in [1, 2, 4, 7, 0]]
        self.assertEqual(np.diff([1, 2, 4, 7, 0]).sum(), rs.accumulative_diff())

        rs = RunningStats()
        values = np.random.uniform(-100, 100, 100)
        [rs.push(n) for n in values]
        self.assertTrue(math.isclose(np.diff(values).sum(), rs.accumulative_diff(), rel_tol=0, abs_tol=0.001))

    def test_slope(self):
        rs = RunningStats()
        self.assertEqual(0, rs.slope())

        rs.push(10)
        self.assertEqual(10, rs.slope())

        rs.push(3)
        self.assertEqual(-7, rs.slope())

    def test_push(self):
        rs = RunningStats()
        rs.push(0)

        self.assertEqual(1, rs.n())
        self.assertEqual(0, rs.curr())
        self.assertEqual(0, rs.last())

        rs.push(10)
        self.assertEqual(2, rs.n())
        self.assertEqual(10, rs.curr())
        self.assertEqual(0, rs.last())

    def test_median(self):
        rs = RunningStats()
        values = np.random.uniform(-100, 100, 10)
        [rs.push(n) for n in values]

        self.assertEqual(np.median(values), rs.median())

    def test_mad(self):
        rs = RunningStats()
        self.assertTrue(math.isnan(rs.median()))

        rs = RunningStats()
        values = [1, 2]
        random.shuffle(values)
        [rs.push(n) for n in values]
        self.assertEqual(1.5, rs.median())
        self.assertEqual(0.5, rs.mad())

        rs = RunningStats()
        values = [2, 2]
        random.shuffle(values)
        [rs.push(n) for n in values]
        self.assertEqual(2, rs.median())
        self.assertEqual(0, rs.mad())

        rs = RunningStats()
        values = [1, 2, 3]
        random.shuffle(values)
        [rs.push(n) for n in values]
        self.assertEqual(2, rs.median())
        self.assertEqual(1, rs.mad())

        rs = RunningStats()
        values = [1, 1, 2, 2, 4, 6, 9]
        random.shuffle(values)
        [rs.push(n) for n in values]
        self.assertEqual(2, rs.median())
        self.assertEqual(1, rs.mad())

    def test_mean(self):
        rs = RunningStats()
        values = np.random.uniform(-100, 100, 10)
        [rs.push(n) for n in values]
        self.assertTrue(math.isclose(values.mean(), rs.mean(), rel_tol=0, abs_tol=0.001))

    def test_exp_mean(self):
        self.fail()

    def test_variance(self):
        rs = RunningStats()
        values = np.random.uniform(-100, 100, 10)
        [rs.push(n) for n in values]
        self.assertTrue(math.isclose(values.var(), rs.variance(), rel_tol=0, abs_tol=0.001), f'{values.var()}, {rs.variance()}',  )

    def test_standard_deviation(self):
        rs = RunningStats()
        values = np.random.uniform(-100, 100, 10)
        [rs.push(n) for n in values]

        self.assertEqual(values.std(), rs.standard_deviation())

    def test_max(self):
        rs = RunningStats()
        values = np.random.uniform(-100, 100, 10)
        [rs.push(n) for n in values]

        self.assertEqual(values.max(), rs.max())

    def test_min(self):
        rs = RunningStats()
        values = np.random.uniform(-100, 100, 10)
        [rs.push(n) for n in values]

        self.assertEqual(values.min(), rs.min())

    def test_t_test(self):
        self.fail()

    def test_serialize(self):
        rs = RunningStats()

        [rs.push(n) for n in [1, 1, 2, 2, 4, 6, 9]]

        for key, value in rs.serialize().items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, numbers.Number)


if __name__ == '__main__':
    unittest.main()
