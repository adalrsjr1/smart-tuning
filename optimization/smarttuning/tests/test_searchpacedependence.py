import unittest
import sys
import yaml
from pprint import pprint
from hyperopt import hp, tpe, fmin, pyll, Trials
from controllers.searchspacemodel import SearchSpaceModel, to_scale

class SearchSpaceDependence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open('../../manifests/search-space/tests/search-space-dependence-full.yaml', 'r') as f:
            cls.searchspace_manifest = yaml.safe_load(f)

    def test_dynamic_hp(self):
        for _ in range(100):

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
            self.assertTrue(5 <= result['b'] <= 10)

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
        for _ in range(1000):
            result = pyll.stochastic.sample(space)
            print(f'xx[{result}]xx')
            self.assertTrue(1 <= result <= 6)


if __name__ == '__main__':
    unittest.main()
