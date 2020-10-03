import unittest
from hyperopt import hp, STATUS_OK
import time
from bayesian import BayesianEngine, BayesianDTO
from concurrent.futures import ThreadPoolExecutor
from hyperopt import fmin,tpe, Trials, STATUS_FAIL, STATUS_OK, space_eval
from hyperopt.fmin import generate_trials_to_calculate


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

    def test_update_trial(self):
        # points = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
        # best = fmin(
        #     fn=lambda space: space["x"] ** 2 + space["y"] ** 2,
        #     space={"x": hp.uniform("x", -10, 10), "y": hp.uniform("y", -10, 10)},
        #     algo=tpe.suggest,
        #     max_evals=10,
        #     points_to_evaluate=points,
        # )
        # assert best["x"] == 0.0
        # assert best["y"] == 0.0
        def fn(space):
            return {
                'loss': space["x"] ** 2 + space["y"] ** 2,
                'status': STATUS_OK,
                # -- store other results like this
                'eval_time': time.time(),
                'classification': None,
                # -- attachments are handled differently
                # https://github.com/hyperopt/hyperopt/wiki/FMin
                # 'attachments':
                #     {'classification': pickle.dumps(classficiation)}
            }

        trials = Trials()
        best = fmin(
            fn=fn,
            space={"x": hp.uniform("x", -10, 10), "y": hp.uniform("y", -10, 10)},
            algo=tpe.suggest,
            max_evals=10,
            trials=trials,
            show_progressbar=False,
        )

        # for trial in trials:
        #     print(trial)
        #
        # print(trials.idxs_vals)
        # print(trials.best_trial)
        # loss = trials.best_trial['result']['loss']
        # best = trials.argmin
        #
        # print(space_eval({"x": hp.uniform("x", -10, 10), "y": hp.uniform("y", -10, 10)}, best))
        # print(best, loss)
        best_idx1 = list(trials.best_trial['misc']['idxs'].values())[0][0]
        for i, trial in enumerate(trials):
            if i == best_idx1:
                trial['result']['loss'] = 100000
                print('## ',trial)
        trials.refresh()
        best = fmin(
            fn=fn,
            space={"x": hp.uniform("x", -10, 10), "y": hp.uniform("y", -10, 10)},
            algo=tpe.suggest,
            max_evals=10,
            trials=trials,
            show_progressbar=False,
        )
        best_idx2 = list(trials.best_trial['misc']['idxs'].values())[0][0]

        self.assertNotEqual(best_idx1, best_idx2)

if __name__ == '__main__':
    unittest.main()
