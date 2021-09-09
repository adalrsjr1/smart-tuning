from __future__ import annotations

import random
import time
import unittest
from datetime import datetime
from unittest import TestCase
from unittest.mock import MagicMock, Mock

import optuna.trial
from optuna.trial import TrialState

from models.configuration import Configuration
from models.instance import Instance
from models.metric2 import Sampler
from models.workload import Workload
from sampler import Metric
from smarttuning2.planner.iteration import TrainingIteration, IterationDriver, ReinforcementIteration, \
    ProbationIteration, TunedIteration, DriverSession, Iteration
from smarttuning2.test_planner.mock_searchspace import MockSearchSpace, MockTrial
from util.stats import RunningStats


class TestIteration(TestCase):

    @staticmethod
    def pmock() -> Instance:
        i = Instance('pmock', '', True, 1, MagicMock(), MagicMock())

        def side_effect_patch(arg):
            print(f'patching: {arg.name}')

        patch = Mock(side_effect=side_effect_patch)
        i.patch_config = patch
        i.patch_current_confi = patch
        return i

    @staticmethod
    def tmock() -> Instance:
        i = Instance('tmock', '', True, 1, MagicMock(), MagicMock())

        def side_effect_patch(arg):
            print(f'patching: {arg.name}')

        patch = Mock(side_effect=side_effect_patch)
        i.patch_config = patch
        i.patch_current_confi = patch
        return i

    @staticmethod
    def training_driver(workload=Workload('workload'),
                        search_space=MockSearchSpace.new(),
                        max_global_iterations: int = 50, max_local_iterations: int = 10,
                        max_reinforcement_iterations: int = 3, max_probation_iterations: int = 3,
                        sampling_interval=0.001, n_sampling_subintervals=3, logging_subinterval=0.2,
                        fail_fast=False) -> IterationDriver:

        production = TestIteration.pmock()
        training = TestIteration.tmock()

        driver = IterationDriver(workload=workload, search_space=search_space, production=production, training=training,
                                 max_global_iterations=max_global_iterations, max_local_iterations=max_local_iterations,
                                 max_reinforcement_iterations=max_reinforcement_iterations,
                                 max_probation_iterations=max_probation_iterations, sampling_interval=sampling_interval,
                                 n_sampling_subintervals=n_sampling_subintervals,
                                 logging_subinterval=logging_subinterval, fail_fast=fail_fast)

        production.set_initial_configuration(
            Configuration.running_config(search_space),
            driver
        )

        driver.curr_workload = MagicMock(return_value=Workload('workload'))

        def save_side_effect(*args, **kwargs):
            return print

        save_mock = Mock(side_effect=save_side_effect)
        driver.save_trace = save_mock

        return driver

    @staticmethod
    def training_iteration(configuration: Configuration = None) -> (TrainingIteration, IterationDriver):
        driver = IterationDriver(workload=Workload('workload'), search_space=MockSearchSpace.new(),
                                 production=TestIteration.pmock(), training=TestIteration.tmock(),
                                 sampling_interval=0.001, n_sampling_subintervals=3, logging_subinterval=0.2,
                                 fail_fast=False)
        return driver.new_training_it(configuration), driver

    def test_init(self):
        return TestIteration.training_iteration()

    def test_sample(self):
        it = TestIteration.training_iteration()[0]
        trial = MockTrial.create_trial()
        self.assertIsInstance(it.sample(trial), Configuration)

    def test_iterate(self):
        return_value1 = 4
        driver = TestIteration.training_driver()
        it = driver.new_training_it()
        it.workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value1)), Metric(to_eval=str(return_value1))))
        it.iterate()

        # initical config has score = 0, median of #2 -> return_value1/2
        # self.assertEqual(it.driver.session().best().final_score(), return_value1 / 2)
        self.assertEqual(it.driver.session().best().final_score(), return_value1)

        return_value2 = 2
        it = driver.new_training_it()
        it.workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value2)), Metric(to_eval=str(return_value2))))
        it.iterate()
        self.assertEqual(it.driver.session().best().final_score(), return_value2)

    def test_reinforcement(self):
        return_value1 = 4
        driver = TestIteration.training_driver()
        it = driver.new_training_it()
        it.workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value1)), Metric(to_eval=str(return_value1))))
        it.iterate()
        print(it.driver.session().best(n=1).name[:3], it.driver.session().best(n=1).debug_stats())
        print([(cfg.name[:3], cfg.debug_stats()) for cfg in it.driver.session().best(n=10)])

        return_value1 = -8
        it = driver.new_training_it()
        it.workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value1)), Metric(to_eval=str(return_value1))))
        it.iterate()
        print(it.driver.session().best(n=1).name[:3], it.driver.session().best(n=1).debug_stats())
        print([(cfg.name[:3], cfg.debug_stats()) for cfg in it.driver.session().best(n=10)])

        r_it = driver.new_reinforcement_it(it.driver.session().best())
        return_value2 = 2
        r_it.workload = MagicMock(return_value=Workload('workload'))
        r_it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value2)), Metric(to_eval=str(return_value2))))
        r_it.iterate()
        # median of [initial=0, return_value1=4, return_value2=2] == 4
        print(it.driver.session().best(n=1).name[:3], it.driver.session().best(n=1).debug_stats())
        print([(cfg.name[:3], cfg.final_score()) for cfg in it.driver.session().best(n=10)])
        self.assertEqual(it.driver.session().best().mean, (return_value1 + return_value2) / 2)

    def test_probation(self):
        return_value1 = 4
        driver = TestIteration.training_driver()
        it = driver.new_training_it()
        it.mostly_workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value1)), Metric(to_eval=str(return_value1))))
        it.iterate()

        p_it = driver.new_probation_it(it.driver.session().best())
        return_value2 = 2
        p_it.mostly_workload = MagicMock(return_value=Workload('workload'))
        p_it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value2)), Metric(to_eval=str(return_value2))))
        p_it.iterate()

        # return_value2 because during probation this value is added twice
        # 1x production and 1x training
        self.assertEqual(it.driver.session().best().final_score(), return_value2)

    def test_tuned(self):
        return_value1 = 4
        driver = TestIteration.training_driver()
        it = driver.new_training_it()
        it.workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value1)), Metric(to_eval=str(return_value1))))
        it.iterate()
        # when training, both the production and training scores are updated
        # initial config has score = 0
        # nursery doesn't
        print(it.driver.session().nursery)
        # self.assertEqual(it.driver.session().best().final_score(), return_value1/2)
        self.assertEqual(it.driver.session().best().final_score(), return_value1)

        print('n:', {c.trial.number: c.trial.value for c in it.driver.session().nursery})
        print('t:', {c.trial.number: c.trial.value for c in it.driver.session().tenured})
        print('s:', {c.number: c.value for c in it.driver.session().study.trials})

        p_it = driver.new_tuned_it(driver.session().best())
        return_value2 = 8
        p_it.workload = MagicMock(return_value=Workload('workload'))
        p_it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value2)), Metric(to_eval=str(return_value2))))
        p_it.iterate()

        print('n:', {c.trial.number: c.trial.value for c in it.driver.session().nursery})
        print('t:', {c.trial.number: c.trial.value for c in it.driver.session().tenured})
        print('s:', {c.number: c.value for c in it.driver.session().study.trials})

        # when tuned, only the production score is updated
        # median of [initial=0, return_value1=4, return_value2=8] == retrun_value1
        print(it.driver.session().best(n=10))
        self.assertEqual(it.driver.session().best().final_score(), (return_value1+return_value2)/2)

    def test_iterate_with_different_workload(self):
        return_value1 = 4
        driver = TestIteration.training_driver()

        it = driver.new_training_it()
        it.mostly_workload = MagicMock(return_value=Workload('workload1'))
        it.workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value1)), Metric(to_eval=str(return_value1))))
        self.assertFalse(it.iterate())

        print([t.state for t in it.driver.session().study.trials])
        # [initial_config, pruned_config, trying_config_again]
        self.assertListEqual([TrialState.COMPLETE, TrialState.PRUNED, TrialState.WAITING],
                             [t.state for t in it.driver.session().study.trials])

        it = driver.new_training_it()
        it.mostly_workload = MagicMock(return_value=Workload('workload1'))
        it.workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value1)), Metric(to_eval=str(return_value1))))
        self.assertFalse(it.iterate())
        self.assertListEqual([TrialState.COMPLETE, TrialState.PRUNED, TrialState.PRUNED, TrialState.WAITING],
                             [t.state for t in it.driver.session().study.trials])

        it = driver.new_training_it()
        it.mostly_workload = MagicMock(return_value=Workload('workload'))
        it.workload = MagicMock(return_value=Workload('workload'))
        it.waiting_for_metrics = MagicMock(
            return_value=(Metric(to_eval=str(return_value1)), Metric(to_eval=str(return_value1))))
        result = it.iterate()
        self.assertTrue(result)
        self.assertListEqual(
            [TrialState.COMPLETE, TrialState.PRUNED, TrialState.PRUNED, TrialState.COMPLETE],
            [t.state for t in it.driver.session().study.trials]
            )

    def test_wait(self):
        driver = TestIteration.training_driver()
        it = driver.new_training_it()
        it.workload = MagicMock(return_value=Workload('workload'))

        driver.production.metrics = lambda : Mock(objective=lambda : 0)
        driver.training.metrics = lambda : Mock(objective=lambda : 0)
        it.iterate()
        now = datetime.utcnow()
        # assert it.training is not None
        # assert it.training.configuration is not None
        # assert it.production is not None
        # assert it.production.configuration is not None
        tm, pm = it.waiting_for_metrics()
        self.assertGreaterEqual((datetime.utcnow() - now).total_seconds(),
                                it.sampling_interval * it.n_sampling_subintervals)

    def test_driver_session_best(self):
        def fn(n):
            session = DriverSession(workload=Workload('test'), driver=MagicMock(), search_space=MockSearchSpace.new(),
                                    production=self.pmock(), training=self.tmock())
            last = 0
            for _ in range(n):
                c = session.ask()
                last = int(time.time())
                c.score = last
                session.tell(c)

            return session.best(), last

        best, last = fn(10)
        self.assertEqual(best.score, last)
        best, last = fn(1)
        self.assertEqual(best.score, last)

        best, last = fn(0)
        self.assertEqual(best, Configuration.empty_config())

        def test_driver_session_promote_nursery(self):
            def fn(n):
                session = DriverSession(workload=Workload('test'), driver=MagicMock(), search_space=MockSearchSpace.new())
                last = 0
                for _ in range(n):
                    c = session.ask()
                    last = int(time.time())
                    c.score = last
                    session.tell(c)
                session.promote_to_tenured()
                self.assertEqual(len(session.nursery) == 0)
                self.assertEqual(len(session.tenured) == 1)
                return session.best(), last

        best, last = fn(10)
        self.assertEqual(best.score, last)
        best, last = fn(1)
        self.assertEqual(best.score, last)

    def test_ask_tell(self):
        driver = TestIteration.training_driver()
        pmetrics = MagicMock(return_value=Metric.zero())
        tmetrics = MagicMock(return_value=Metric.zero())
        driver.production.metrics = pmetrics
        driver.training.metrics = tmetrics

        c: Configuration = driver.session().ask()
        c.score = -100
        self.assertFalse(len(c.trial.params) == 0, 'trial.params shouldn\'t be empty')
        driver.session().tell(c)
        self.assertEqual(driver.session().best(), c)

    def test_best_nursery(self):
        driver = TestIteration.training_driver()

        best_value = float('inf')
        best_name = ''

        for i in range(10):
            score = random.randint(-1000, 0)
            for _ in range(3):
                c = Configuration(trial=optuna.trial.FixedTrial({i: score}, number=0), data={i: score})
                c.score = score
                if c.score < best_value:
                    best_value = c.score
                    best_name = c.name

                driver.session().update_heap(driver.session().nursery, c)
        best_c = driver.session().best(n=3)
        print(driver.session().nursery)
        c = best_c[0]
        self.assertEqual(best_value, c.score, msg=f'wrong config, expected (cfg:{best_name} score={best_value})'
                                                  f'given: (cfg: {c.name}, score={c.score})')

        self.assertNotEqual(best_c[0].name, best_c[1].name)
        self.assertLessEqual(best_c[0].score, best_c[1].score)
        self.assertNotEqual(best_c[1].name, best_c[2].name)
        self.assertLessEqual(best_c[1].score, best_c[2].score)
        self.assertNotEqual(best_c[0].name, best_c[2].name)
        self.assertLessEqual(best_c[0].score, best_c[2].score)

    def test_best_tenured(self):
        driver = TestIteration.training_driver()

        best_value = float('inf')
        best_name = ''

        for i in range(10):
            if i % 2 == 0:
                score = random.randint(-500, 0)
            else:
                score = random.randint(-1000, 501)
            for _ in range(3):
                c = Configuration(trial=optuna.trial.FixedTrial({i: score}, number=0), data={i: score})
                c.score = score
                if c.score < best_value:
                    best_value = c.score
                    best_name = c.name

                if i % 2 == 0:
                    driver.session().update_heap(driver.session().nursery, c)
                else:
                    driver.session().update_heap(driver.session().tenured, c)
        best_c = driver.session().best(n=3)

        c = best_c[0]
        self.assertEqual(best_value, c.score, msg=f'wrong config, expected (cfg:{best_name} score={best_value})'
                                                  f'given: (cfg: {c.name}, score={c.score})')

        self.assertNotEqual(best_c[0].name, best_c[1].name)
        self.assertLess(best_c[0].score, best_c[1].score)
        self.assertNotEqual(best_c[1].name, best_c[2].name)
        self.assertLess(best_c[1].score, best_c[2].score)
        self.assertNotEqual(best_c[0].name, best_c[2].name)
        self.assertLess(best_c[0].score, best_c[2].score)

        for c in best_c:
            driver.session().update_heap(driver.session().tenured, c)

        best_c = driver.session().best(n=3)

        c = best_c[0]
        self.assertEqual(best_value, c.score, msg=f'wrong config, expected (cfg:{best_name} score={best_value})'
                                                  f'given: (cfg: {c.name}, score={c.score})')

        self.assertNotEqual(best_c[0].name, best_c[1].name)
        self.assertLess(best_c[0].score, best_c[1].score)
        self.assertNotEqual(best_c[1].name, best_c[2].name)
        self.assertLess(best_c[1].score, best_c[2].score)
        self.assertNotEqual(best_c[0].name, best_c[2].name)
        self.assertLess(best_c[0].score, best_c[2].score)

    def test_driver_progressions(self):
        def iterations(max_g=0,
                       max_l=0,
                       max_r=0,
                       max_p=0,
                       trace=None,
                       tscore=0, pscore=0):
            driver = TestIteration.training_driver(max_global_iterations=max_g,
                                                   max_local_iterations=max_l,
                                                   max_reinforcement_iterations=max_r,
                                                   max_probation_iterations=max_p)

            class MetricSideEffect:
                def __init__(self, t_increment, p_increment):
                    self.t_increment = t_increment
                    self.p_increment = p_increment
                    self.t_running_stats = RunningStats()
                    self.t_running_stats.push(0)
                    self.p_running_stats = RunningStats()
                    self.p_running_stats.push(0)

                def t_side_effect(self):
                    self.t_running_stats.push(self.t_increment)
                    m = Metric(to_eval=str(self.t_running_stats.mean()))
                    # print(f't: {m}')
                    return m

                def p_side_effect(self):
                    self.p_running_stats.push(self.p_increment)
                    m = Metric(to_eval=str(self.p_running_stats.mean()))
                    # print(f'p: {m}')
                    return m

            pmock = Mock()
            sideeffects = MetricSideEffect(t_increment=tscore, p_increment=pscore)
            pmock.side_effect = sideeffects.p_side_effect

            tmock = Mock()
            tmock.side_effect = sideeffects.t_side_effect

            driver.production.metrics = pmock
            driver.training.metrics = tmock

            self.assertEqual(driver.max_global_iterations, max_g)
            self.assertEqual(driver.max_local_iterations, max_l)
            self.assertEqual(driver.max_reinforcement_iterations, max_r)
            self.assertEqual(driver.max_probation_iterations, max_p)

            # creating trace
            if trace is None:
                trace = []

            self.assertFalse(max_g > 0 and len(trace) == 0, msg=f'#iterations != #expected')
            # for i, expected in enumerate(new_l):
            for i, expected in enumerate(trace):
                it: Iteration
                it = next(driver)
                print(f'p:{it.production.configuration.final_score()}, t:{it.training.configuration.final_score()}')

                print('expected:', expected.__name__, ', actual:', type(it).__name__)
                self.assertEqual(expected, type(it), msg=f'failed at {i}')

            self.assertEqual(driver.global_iteration, len(trace), msg=f'#iterations != #expected')

        # progress to reinforcement
        iterations(max_g=3,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration, ReinforcementIteration, ProbationIteration, TrainingIteration],
                   pscore=-0.1, tscore=-1)

        # progress to probation
        iterations(max_g=4,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration, ReinforcementIteration, ProbationIteration, TrainingIteration],
                   pscore=-0.1, tscore=-1)

        # progress to probation
        iterations(max_g=2,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration, ReinforcementIteration, ProbationIteration, TrainingIteration],
                   pscore=-0.1, tscore=-1)

        # don't stop during reinforcement
        iterations(max_g=1,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration],
                   pscore=-0.1, tscore=-1)

        # don't stop during probation
        iterations(max_g=2,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration, ReinforcementIteration, ProbationIteration, TrainingIteration],
                   pscore=-0.1, tscore=-1)

        # don't stop during probation
        iterations(max_g=80,
                   max_l=10,
                   max_r=3,
                   max_p=3,
                   trace=([TrainingIteration] * 10 + [ReinforcementIteration] * 3 + [ProbationIteration] * 3) * 5 + [
                       TrainingIteration],
                   pscore=-0.1, tscore=-1)

        # don't stop during probation
        iterations(max_g=16,
                   max_l=10,
                   max_r=3,
                   max_p=3,
                   trace=([TrainingIteration] * 10 + [ReinforcementIteration] * 3 + [ProbationIteration] * 3) * 1 + [
                       TrainingIteration] + [TunedIteration] * 2,
                   pscore=-0.1, tscore=-1)

        # don't stop during probation
        iterations(max_g=80,
                   max_l=10,
                   max_r=3,
                   max_p=3,
                   trace=([TrainingIteration] * 10 + [ReinforcementIteration] * 3 + [ProbationIteration] * 3) * 5 + [
                       TrainingIteration] + [TunedIteration] * 20,
                   pscore=-0.1, tscore=-1)

    def test_driver_progressions_doesnt_progress_to_reinforcement(self):
        def iterations(max_g=0,
                       max_l=0,
                       max_r=0,
                       max_p=0,
                       trace=None,
                       tscore=0, pscore=0, use_driver: bool = False):
            driver = TestIteration.training_driver(max_global_iterations=max_g,
                                                   max_local_iterations=max_l,
                                                   max_reinforcement_iterations=max_r,
                                                   max_probation_iterations=max_p)

            class MetricSideEffect:
                def __init__(self, t_increment, p_increment, driver=None):
                    self.driver = driver
                    self._t_increment = t_increment
                    self._p_increment = p_increment
                    self.t_running_stats = RunningStats()
                    self.t_running_stats.push(0)
                    self.p_running_stats = RunningStats()
                    self.p_running_stats.push(0)

                @staticmethod
                def __call_or_return(to_test, driver=None):
                    if callable(to_test):
                        if driver:
                            return to_test(driver)
                        else:
                            return to_test()
                    else:
                        return to_test

                @property
                def t_increment(self):
                    return MetricSideEffect.__call_or_return(self._t_increment, self.driver)

                @property
                def p_increment(self):
                    return MetricSideEffect.__call_or_return(self._p_increment, self.driver)

                def t_side_effect(self):
                    self.t_running_stats.push(self.t_increment)
                    m = Metric(to_eval=str(self.t_running_stats.mean()))
                    # print(f't: {m}')
                    return m

                def p_side_effect(self):
                    self.p_running_stats.push(self.p_increment)
                    m = Metric(to_eval=str(self.p_running_stats.mean()))
                    # print(f'p: {m}')
                    return m

            pmock = Mock()
            sideeffects = MetricSideEffect(t_increment=tscore, p_increment=pscore,
                                           driver=driver if use_driver else None)
            pmock.side_effect = sideeffects.p_side_effect

            tmock = Mock()
            tmock.side_effect = sideeffects.t_side_effect

            driver.production.metrics = pmock
            driver.training.metrics = tmock

            self.assertEqual(driver.max_global_iterations, max_g)
            self.assertEqual(driver.max_local_iterations, max_l)
            self.assertEqual(driver.max_reinforcement_iterations, max_r)
            self.assertEqual(driver.max_probation_iterations, max_p)

            # creating trace
            if trace is None:
                trace = []

            self.assertFalse(max_g > 0 and len(trace) == 0, msg=f'#iterations != #expected')
            # for i, expected in enumerate(new_l):
            for i, expected in enumerate(trace):
                it: Iteration
                it = next(driver)
                print(f'p:{it.production.configuration.final_score()}, t:{it.training.configuration.final_score()}')

                print('expected:', expected.__name__, ', actual:', type(it).__name__)
                self.assertEqual(expected, type(it), msg=f'failed at {i}')

            self.assertEqual(driver.global_iteration, len(trace), msg=f'#iterations != #expected')

        # doenst progress to reinforcement
        iterations(max_g=3,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration] * 3,
                   pscore=-1, tscore=-0.1)

        # tuning ends and applies only already tuned configs that is the default if has none previously tuned
        iterations(max_g=3,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration] * 3 + [TunedIteration] * 3,
                   pscore=-1, tscore=-0.1)

        # doesnt progress for a 2nd reinf/probation cycle
        def pscore_gen(driver: IterationDriver = None):
            if driver:
                print('p >>>: ', driver.global_iteration)

                if driver.global_iteration >= 3:
                    return -1000
                else:
                    return -0.1

            return -0.1

        def tscore_gen(driver: IterationDriver = None):
            if driver:
                print('t >>>: ', driver.global_iteration)

            if driver.global_iteration >= 3:
                return -0.0001
            else:
                return -1

            return -1

        iterations(max_g=6,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration, ReinforcementIteration, ProbationIteration,
                          TrainingIteration, TrainingIteration, TrainingIteration, TunedIteration, TunedIteration],
                   pscore=pscore_gen, tscore=tscore_gen, use_driver=True)

        iterations(max_g=10,
                   max_l=1,
                   max_r=1,
                   max_p=0,
                   trace=[TrainingIteration] * 3,
                   pscore=-1, tscore=-0.1)

        with self.assertRaises(AssertionError):
            iterations(max_g=3,
                       max_l=1,
                       max_r=1,
                       max_p=1,
                       trace=[TrainingIteration, ReinforcementIteration, ProbationIteration, TrainingIteration],
                       pscore=-1, tscore=-0.1)

    def test_driver_progressions_doesnt_progress_to_probation(self):
        def iterations(max_g=0,
                       max_l=0,
                       max_r=0,
                       max_p=0,
                       trace=None,
                       tscore=0, pscore=0, use_driver: bool = False):
            driver = TestIteration.training_driver(max_global_iterations=max_g,
                                                   max_local_iterations=max_l,
                                                   max_reinforcement_iterations=max_r,
                                                   max_probation_iterations=max_p)

            class MetricSideEffect:
                def __init__(self, t_increment, p_increment, driver=None):
                    self.driver = driver
                    self._t_increment = t_increment
                    self._p_increment = p_increment
                    self.t_running_stats = RunningStats()
                    self.t_running_stats.push(0)
                    self.p_running_stats = RunningStats()
                    self.p_running_stats.push(0)

                @staticmethod
                def __call_or_return(to_test, driver=None):
                    if callable(to_test):
                        if driver:
                            return to_test(driver)
                        else:
                            return to_test()
                    else:
                        return to_test

                @property
                def t_increment(self):
                    return MetricSideEffect.__call_or_return(self._t_increment, self.driver)

                @property
                def p_increment(self):
                    return MetricSideEffect.__call_or_return(self._p_increment, self.driver)

                def t_side_effect(self):
                    self.t_running_stats.push(self.t_increment)
                    m = Metric(to_eval=str(self.t_running_stats.mean()))
                    # print(f't: {m}')
                    return m

                def p_side_effect(self):
                    self.p_running_stats.push(self.p_increment)
                    m = Metric(to_eval=str(self.p_running_stats.mean()))
                    # print(f'p: {m}')
                    return m

            pmock = Mock()
            sideeffects = MetricSideEffect(t_increment=tscore, p_increment=pscore,
                                           driver=driver if use_driver else None)
            pmock.side_effect = sideeffects.p_side_effect

            tmock = Mock()
            tmock.side_effect = sideeffects.t_side_effect

            def mean(self):
                return self.mean

            Configuration.final_score = mean
            driver.production.metrics = pmock
            driver.training.metrics = tmock

            self.assertEqual(driver.max_global_iterations, max_g)
            self.assertEqual(driver.max_local_iterations, max_l)
            self.assertEqual(driver.max_reinforcement_iterations, max_r)
            self.assertEqual(driver.max_probation_iterations, max_p)

            # creating trace
            if trace is None:
                trace = []

            self.assertFalse(max_g > 0 and len(trace) == 0, msg=f'#iterations != #expected')
            # for i, expected in enumerate(new_l):
            for i, expected in enumerate(trace):
                it: Iteration
                it = next(driver)
                print(f'p:{it.production.configuration.final_score()}, t:{it.training.configuration.final_score()}')

                print('expected:', expected.__name__, ', actual:', type(it).__name__)
                self.assertEqual(expected, type(it), msg=f'failed at {i}')

            self.assertEqual(driver.global_iteration, len(trace), msg=f'#iterations != #expected')

        # doenst progress to probation
        def pscore_gen(driver: IterationDriver = None):
            if driver:
                print('p >>>: ', driver.global_iteration)

                if driver.global_iteration > 0:
                    return -1000000
                else:
                    return -0.1

            return -0.1

        def tscore_gen(driver: IterationDriver = None):
            if driver:
                print('t >>>: ', driver.global_iteration)

            if driver.global_iteration > 0:
                return -0.0000001
            else:
                return -1

            return -1

        iterations(max_g=3,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration, ReinforcementIteration, TrainingIteration, TunedIteration],
                   pscore=pscore_gen, tscore=tscore_gen, use_driver=True)

        # doenst progress to probation at 2nd cycle
        def pscore_gen(driver: IterationDriver = None):
            if driver:
                print('p >>>: ', driver.global_iteration)

                if driver.global_iteration > 3:
                    return -1000000
                else:
                    return -0.1

            return -0.1

        def tscore_gen(driver: IterationDriver = None):
            if driver:
                print('t >>>: ', driver.global_iteration)

            if driver.global_iteration > 3:
                return -0.0000001
            else:
                return -1

            return -1

        iterations(max_g=6,
                   max_l=1,
                   max_r=1,
                   max_p=1,
                   trace=[TrainingIteration, ReinforcementIteration, ProbationIteration,
                          TrainingIteration, ReinforcementIteration, TrainingIteration, TunedIteration],
                   pscore=pscore_gen, tscore=tscore_gen, use_driver=True)

    def test_interleaved_workload(self):
        driver1 = TestIteration.training_driver(max_global_iterations=3,
                                               max_local_iterations=1,
                                               max_reinforcement_iterations=1,
                                               max_probation_iterations=1)

        driver2 = TestIteration.training_driver(max_global_iterations=3,
                                               max_local_iterations=1,
                                               max_reinforcement_iterations=1,
                                               max_probation_iterations=1)

        c1 = driver1.session().ask()
        c1.final_score = lambda: 0
        driver1.session().tell(c1)

        c2 = driver1.session().ask()
        c2.final_score = lambda: -10
        driver1.session().tell(c2)

        c3 = driver1.session().ask()
        c3.final_score = lambda: -100
        driver2.session().add(c3)

        _c3 = driver2.session().best_nursery(3)[0]
        self.assertEqual(c3.name, _c3.name)
        self.assertIsNot(c3, _c3)
        self.assertNotEqual(c3.trial.number, _c3.trial.number)
        self.assertEqual(c3.trial.number, len(driver1.session().study.trials)-1)
        self.assertEqual(_c3.trial.number, len(driver2.session().study.trials)-1)

    def test_driver_skip_reinforcement(self):
        def iterations(max_g=0,
                       max_l=0,
                       max_r=0,
                       max_p=0,
                       trace=None,
                       tscore=0, pscore=0):
            driver = TestIteration.training_driver(max_global_iterations=max_g,
                                                   max_local_iterations=max_l,
                                                   max_reinforcement_iterations=max_r,
                                                   max_probation_iterations=max_p)

            class MetricSideEffect:
                def __init__(self, t_increment, p_increment):
                    self.t_increment = t_increment
                    self.p_increment = p_increment
                    self.t_running_stats = RunningStats()
                    self.t_running_stats.push(0)
                    self.p_running_stats = RunningStats()
                    self.p_running_stats.push(0)

                def t_side_effect(self):
                    self.t_running_stats.push(self.t_increment)
                    m = Metric(to_eval=str(self.t_running_stats.mean()))
                    # print(f't: {m}')
                    return m

                def p_side_effect(self):
                    self.p_running_stats.push(self.p_increment)
                    m = Metric(to_eval=str(self.p_running_stats.mean()))
                    # print(f'p: {m}')
                    return m

            pmock = Mock()
            sideeffects = MetricSideEffect(t_increment=tscore, p_increment=pscore)
            pmock.side_effect = sideeffects.p_side_effect

            tmock = Mock()
            tmock.side_effect = sideeffects.t_side_effect

            driver.production.metrics = pmock
            driver.training.metrics = tmock

            self.assertEqual(driver.max_global_iterations, max_g)
            self.assertEqual(driver.max_local_iterations, max_l)
            self.assertEqual(driver.max_reinforcement_iterations, max_r)
            self.assertEqual(driver.max_probation_iterations, max_p)

            # creating trace
            if trace is None:
                trace = []

            self.assertFalse(max_g > 0 and len(trace) == 0, msg=f'#iterations != #expected')
            # for i, expected in enumerate(new_l):
            for i, expected in enumerate(trace):
                it: Iteration
                it = next(driver)
                print(f'p:{it.production.configuration.final_score()}, t:{it.training.configuration.final_score()}')

                print('expected:', expected.__name__, ', actual:', type(it).__name__)
                self.assertEqual(expected, type(it), msg=f'failed at {i}')

            self.assertEqual(driver.global_iteration, len(trace), msg=f'#iterations != #expected')

        # progress to reinforcement but skip it
        iterations(max_g=3,
                   max_l=1,
                   max_r=0,
                   max_p=1,
                   trace=[TrainingIteration, TrainingIteration, ProbationIteration, TrainingIteration],
                   pscore=-0.1, tscore=-1)

        # progress to probation but skip it
        iterations(max_g=3,
                   max_l=1,
                   max_r=1,
                   max_p=0,
                   # the 2nd ReinforcementIteration fallback into handle_reset() that just pass through all logic
                   # and reset the counters
                   trace=[TrainingIteration, ReinforcementIteration, ReinforcementIteration, TrainingIteration],
                   pscore=-0.1, tscore=-1)

    def test_driver_serialize(self):
        driver = TestIteration.training_driver(max_global_iterations=1,
                                      max_local_iterations=1,
                                      max_reinforcement_iterations=1,
                                      max_probation_iterations=1)

        for key, value in driver.serialize(driver.session()).items():
            self.assertIsNotNone(value, f'{key}:{value}')



if __name__ == '__main__':
    unittest.main()
