import unittest
import numpy as np
from seqkmeans import __resize__

class ResizeTestCase(unittest.TestCase):
    def test_resize_u_gt_v(self):
        u = np.array([3, 1, 2])
        ul = ['a', 'b', 'c']
        v = np.array([2, 7])
        vl = ['b', 'd']

        u, v, _, _ = __resize__(u, ul, v, vl)
        self.assertListEqual([3, 1, 2, 0], list(u))
        self.assertListEqual([0, 2, 0, 7], list(v))

    def test_resize_u_eq_v(self):
        u = [3, 1, 2]
        ul = ['a', 'b', 'c']
        v = [2, 7, 9]
        vl = ['b', 'd', 'e']

        u, v, _, _ = __resize__(u, ul, v, vl)
        self.assertListEqual([3, 1, 2, 0, 0], list(u))
        self.assertListEqual([0, 2, 0, 7, 9], list(v))

    def test_resize_eq_values(self):
        u = [1, 1, 1]
        ul = ['a', 'b', 'c']
        v = [1, 1, 1]
        vl = ['b', 'd', 'e']

        u, v, _, _ = __resize__(u, ul, v, vl)
        self.assertListEqual([1, 1, 1, 0, 0], list(u))
        self.assertListEqual([0, 1, 0, 1, 1], list(v))

    def test_resize_empty(self):
        u = []
        ul = []
        v = []
        vl = []

        u, v, _, _ = __resize__(u, ul, v, vl)
        self.assertListEqual([], list(u))
        self.assertListEqual([], list(v))

    def test_resize_one_empty(self):
        u = [1]
        ul = ['a']
        v = []
        vl = []

        u, v, a, b = __resize__(u, ul, v, vl)

        self.assertListEqual([1], list(u))
        self.assertListEqual([0], list(v))

    def test_resize_one_oposite_empty(self):
        v = [1]
        vl = ['a']
        u = []
        ul = []

        u, v, a, b = __resize__(u, ul, v, vl)

        self.assertListEqual([1], list(v))
        self.assertListEqual([0], list(u))

    def test_resize_value_neq_label(self):
        v = []
        vl = ['a']
        u = []
        ul = ['b']

        u, v, a, b = __resize__(u, ul, v, vl)

        self.assertListEqual([0,0], list(v))
        self.assertListEqual([0,0], list(u))


if __name__ == '__main__':
    unittest.main()
