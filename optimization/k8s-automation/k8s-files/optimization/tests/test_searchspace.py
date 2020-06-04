import unittest
import hyperopt

from configsampler import SearchSpace, load_search_space

class TestSearchSpace(unittest.TestCase):
    def test_load_searchspace(self):
        searchSpace = load_search_space('search_space.json')
        self.assertTrue(searchSpace is not None)

        for _  in range(10):
            print(hyperopt.pyll.stochastic.do_sample(searchSpace.search_space()))


if __name__ == '__main__':
    unittest.main()
