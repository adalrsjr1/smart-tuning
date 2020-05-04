import unittest
from seqkmeans import __resize__

class ResizeTestCase(unittest.TestCase):
    def test_resize_u_gt_v(self):
        u = [3, 1, 2]
        ul = ['a', 'b', 'c']
        v = [2, 7]
        vl = ['b', 'd']

        u, v = __resize__(u, ul, v, vl)
        self.assertListEqual([3, 1, 2, 0], u)
        self.assertListEqual([0, 2, 0, 7], v)

    def test_resize_u_eq_v(self):
        u = [3, 1, 2]
        ul = ['a', 'b', 'c']
        v = [2, 7, 9]
        vl = ['b', 'd', 'e']

        u, v = __resize__(u, ul, v, vl)
        self.assertListEqual([3, 1, 2, 0, 0], u)
        self.assertListEqual([0, 2, 0, 7, 9], v)

    def test_resize_empty(self):
        u = []
        ul = []
        v = []
        vl = []

        u, v = __resize__(u, ul, v, vl)
        self.assertListEqual([], u)
        self.assertListEqual([], v)


if __name__ == '__main__':
    unittest.main()
