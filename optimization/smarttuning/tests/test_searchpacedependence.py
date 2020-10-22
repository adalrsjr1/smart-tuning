import unittest
import sys
import yaml
from pprint import pprint
from hyperopt import hp, tpe, fmin, pyll, Trials
from controllers.searchspacemodel import SearchSpaceModel, to_scale, normalize

class SearchSpaceDependence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open('../../manifests/search-space/tests/search-space-dependence-full.yaml', 'r') as f:
            cls.searchspace_manifest = yaml.safe_load(f)

    def test_scaling_y_within_x(self):
        y = hp.uniform('y', 2, 3)
        x = hp.uniform('x', 1, 4)
        for i in range(1000):
            value = to_scale(x1=1, y1=2, x2=4, y2=3, k=x)
            result = pyll.stochastic.sample(value)
            self.assertTrue(2 <= result <= 3, f'[{i}] 2 <= {result} <= 3 == False')

    def test_scaling_x_within_y(self):
        y = hp.uniform('y', 1, 4)
        x = hp.uniform('x', 2, 3)
        for i in range(1000):
            value = to_scale(x1=2, y1=1, x2=3, y2=4, k=x)
            result = pyll.stochastic.sample(value)
            self.assertTrue(1 <= result <= 4, f'[{i}] 1 <= {result} <= 4 == False')

    def test_scaling_left_y_within_x(self):
        y = hp.uniform('y', 2, 5)
        x = hp.uniform('x', 2, 4)
        for i in range(1000):
            value = to_scale(2, 2, 4, 5, x)
            result = pyll.stochastic.sample(value)
            self.assertTrue(2 <= result <= 5, f'[{i}] 2 <= {result} <= 5 == False')

    def test_scaling_right_y_within_x(self):
        y = hp.uniform('y', 1, 4)
        x = hp.uniform('x', 2, 4)
        for _ in range(1000):
            value = to_scale(2, 1, 4, 4, x)
            result = pyll.stochastic.sample(value)
            self.assertTrue(1 <= result <= 4, f'1 <= {result} <= 4 == False')

    def test_lower(self):
        y = hp.uniform('y', 100, 200)
        x = hp.uniform('x', 100, 200)
        for _ in range(1000):
            yy = pyll.stochastic.sample(y)
            value = to_scale(100, yy, 200, 200, x)
            result = pyll.stochastic.sample(value)
            self.assertTrue(yy <= result <= 200, f'{yy} <= {result} <= 200 == False')

    def test_upper(self):
        y = hp.uniform('y', 100, 200)
        x = hp.uniform('x', 100, 200)
        for _ in range(1000):
            yy = pyll.stochastic.sample(y)
            value = to_scale(100, 100, 200, yy, x)
            result = pyll.stochastic.sample(value)
            self.assertTrue(100 <= result <= yy, f'100 <= {result} <= {yy} == False')

    def test_dynamic_hp(self):
        for i in range(1000):

            a = hp.uniform('a', 5, 10)
            # b = hp.uniform('b', 5, a)
            b = hp.uniform('b', 5, 20)
            # b.pprint(ofile=sys.stdout)
            lower = 5
            upper = a
            value = to_scale(5, 5, 20, 10, b)
            space = {'a': a, 'b': value, 'b0': b}
            # space = {'a':  a, 'b':hp.uniform('b', 1, a)}
            # space = {'a': a, 'b':hp.uniform('b', 0, a)}
            # space = {'a': a, 'b':hp.uniform('b', a, 20)}
            result = pyll.stochastic.sample(space)
            self.assertTrue(5 <= result['b'] <= 10, f'[{i}] 5 <= {result} <= 10 == False')

    def test_load_searchspace(self):
        """
        - name: "replicas"
          lower:
            value: 1
          upper:
            value: 3
        - name: "upstream"
          real: true
          lower:
            value: 4
          upper:
            value: 6
        - name: "downstream"
          real: true
          lower:
            value: 5
            dependsOn: "replicas"
          upper:
            value: 10
            dependsOn: "upstream"
        """
        model = SearchSpaceModel(self.searchspace_manifest)
        space = model.search_space()['acmeair-nginx-test-service']['downstream']
        for i in range(1000):
            result = pyll.stochastic.sample(space)
            self.assertTrue(1 <= result <= 6, f'[{i}] 1 <= {result} <=6 == False')

    def test_load_searchspace2(self):
        """
        - name: "a"
          lower:
            value: 100
          upper:
            value: 200
        - name: "b"
          lower:
            value: 100
            dependsOn: "a"
          upper:
            value: 200
        """
        model = SearchSpaceModel(self.searchspace_manifest)
        space = model.search_space()['acmeair-nginx-test-service']
        for i in range(1000):
            result = pyll.stochastic.sample(space)
            a = result['a']
            b = result['b']
            print(result)
            self.assertTrue(a <= b <= 200, f'[{i}] {a} <= {b} <= 200 == False')


    def test_lower(self):
        for i in range(1000):
            uniform = lambda l, u : hp.uniform('a', l, u)
            a = uniform(5, 10)
            b = uniform(10, 20)
            space = {
                'a': a,
                'b': b,
                'c': (a+20)/2,
                'c0': normalize(a, 10, 5)
            }
            result = pyll.stochastic.sample(space)
            print(result)
            self.assertTrue(result['a'] <= result['c'] <= 20, f'[{i}] {result["a"]} <= {result["c"]} <= 20 == False')

    def test_upper(self):
        for i in range(1000):
            a = hp.uniform('a', 5, 10)
            b = hp.uniform('b', 10, 20)
            space = {
                'a': a,
                'b': b,
                'c': (5 + b)/2,
                'c0': normalize(b, 20, 10)
            }
            result = pyll.stochastic.sample(space)
            print(result)
            self.assertTrue(5 <= result['c'] <= result['b'], f'[{i}] 5 <= {result["c"]} <= {result["b"]} == False')

    def test_lower_upper(self):
        for i in range(1000):
            a = {'a': hp.uniform('a', 5, 10)}
            b = {'b': hp.uniform('b', 10, 20)}
            space = {
                'a': a['a'],
                'b': b['b'],
                'c': (a['a'] + b['b'])/2,
                'c_a': normalize(a['a'], 20, 5),
                'c_b': normalize(b['b'], 20, 5),
            }
            print(space['a'], space['c'])
            result = pyll.stochastic.sample(space)
            print(result)
            self.assertTrue(result['a'] <= result['c'] <= result['b'], f'[{i}] {result["a"]} <= {result["c"]} <= {result["b"]} == False')

if __name__ == '__main__':
    unittest.main()
