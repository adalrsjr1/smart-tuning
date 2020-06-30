import unittest
from hyperopt import hp, STATUS_OK
from bayesian import BayesianEngine, BayesianDTO
from concurrent.futures import ThreadPoolExecutor


class TestBayesianEngine(unittest.TestCase):

    executor = ThreadPoolExecutor()

    def test_in_out_objective(self):
        b = BayesianEngine(name='test')
        dto = BayesianDTO(classification='test')
        future = TestBayesianEngine.executor.submit(b.objective, {'a':1, 'b':2})
        b.put(dto)
        self.assertTrue(set({'loss':0, 'status':STATUS_OK, 'classification':'test'}.items()) <= set(future.result().items()))


    def test_loop(self):
        space = {'x':hp.uniform('x', -10, 10), 'o':hp.choice('o', ['x','y','z'])}
        b = BayesianEngine(name='test', space=space, max_evals=100)
        dto = BayesianDTO(classification='test')
        l = []
        for _ in range(100):
            l.append(b.get())
            b.put(dto)

        self.assertEqual(len(l), 100)

        print(b.best_so_far())

    def test_multiple_loops(self):
        for _ in range(10):
            self.test_loop()



if __name__ == '__main__':
    unittest.main()
