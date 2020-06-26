import unittest
from hyperopt import fmin, tpe, hp, rand, Trials, STATUS_OK, STATUS_FAIL, space_eval
from updateconfig.bayesian import BayesianEngine, BayesianDTO
from concurrent.futures import ThreadPoolExecutor


class TestBayesianEngine(unittest.TestCase):

    executor = ThreadPoolExecutor()

    def test_in_out_objective(self):
        b = BayesianEngine(id='test')
        dto = BayesianDTO(classification='test')
        future = TestBayesianEngine.executor.submit(b.objective, {'a':1, 'b':2})
        b.put(dto)
        self.assertTrue(set({'loss':0, 'status':STATUS_OK, 'classification':'test'}.items()) <= set(future.result().items()))


    def test_loop(self):
        space = {'x':hp.uniform('x', -10, 10), 'o':hp.choice('o', ['x','y','z'])}
        b = BayesianEngine(id='test', space=space, max_evals=10)
        dto = BayesianDTO(classification='test')
        l = []
        for _ in range(10):
            l.append(b.get())
            b.put(dto)

        self.assertEqual(len(l), 10)

        print(b.best_so_far())





if __name__ == '__main__':
    unittest.main()
