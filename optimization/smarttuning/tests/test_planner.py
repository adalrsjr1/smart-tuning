import copy
import os
import random
import sys
import time
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import optuna
from hyperopt import Trials, STATUS_OK
from optuna.distributions import UniformDistribution
from optuna.samplers import RandomSampler

from controllers.planner import Configuration
from controllers.planner import Planner
from controllers.searchspacemodel import SearchSpaceModel
from models.instance import Instance
from models.smartttuningtrials import SmartTuningTrials
from sampler import Metric


def metric(production=True):
    # pmetrics = sampler_pmetric()
    # tmetrics = sampler_tmetric()
    counter = 0

    def anonymous():
        # raw_metric = next(pmetrics) if production else next(tmetrics)
        global counter
        counter += 1
        return Metric(
            name='',
            # cpu=raw_metric['cpu'],
            # memory=raw_metric['memory'],
            # throughput=raw_metric['throughput'],
            # process_time=raw_metric['process_time'],
            # errors=raw_metric['errors'],
            # to_eval=f'{random.uniform(-random.uniform(500,300), -random.uniform(300,100))}'
            to_eval=f'{counter}'
        )

    return anonymous


def search_space(study):
    ctx = SearchSpaceModel({
        "spec": {
            "deployment": "acmeair-service",
            "manifests": [
                {
                    "name": "manifest-name",
                    "type": "configMap"
                }
            ],
            "namespace": "default",
            "service": "fake-svc"
        },
        "data": [
            {
                "filename": "",
                "name": "manifest-name",
                "tunables": {

                    "number": [
                        {
                            "name": f"{c}",
                            "real": True,
                            "lower": {
                                "dependsOn": "",
                                "value": 0
                            },
                            "upper": {
                                "dependsOn": "",
                                "value": 101
                            }
                        } for c in 'abcdefghijklmnopqrstuvwxxyz'
                    ]
                }
            },
        ],
    }, study)
    return ctx


def fake_trial(x: int) -> optuna.trial.FrozenTrial:
    return optuna.trial.create_trial(
        params={c: ord(c)+x for c in 'abcdefghijklmnopqrstuvwxxyz'},
        distributions={'x': UniformDistribution(low=0, high=x + 1)},
        value= [(ord(c)+x)**(26-i) for i, c in enumerate('abcdefghijklmnopqrstuvwxxyz')]
    )


def configurations(study: optuna.Study, strials: SmartTuningTrials):
    configs = []
    for trial in study.get_trials(deepcopy=False):
        configs.append(Configuration(trial, strials))

    return configs


def smarttuning_trials():
    return SmartTuningTrials(space=search_space(study=optuna.create_study(sampler=RandomSampler)))

def get_from_engine(trials: SmartTuningTrials, ctx: SearchSpaceModel):
    def anonymous():
        trial = trials.new_trial_entry({c: random.uniform(10, 100) for c in 'abcdefghijklmnopqrstuvwxxyz'}, loss= random.uniform(100,300))
        c = Configuration(trial=trial, trials=trials)
        trials.add_new_configuration(c)
        return c

    return anonymous

def get_current_config(trials: SmartTuningTrials):
    def anonymous():
        return {c: random.uniform(10, 100) for c in 'abcdefghijklmnopqrstuvwxxyz'}

    return anonymous

counter = 0

c1 = 0
c2 = 0


def wait_for_metrics(planner: Planner):
    def anonymous(*kwargs):
        return Metric(to_eval=f'{random.uniform(-300, -100)}'), Metric(to_eval=f'{random.uniform(-300, -100)}')

    return anonymous


class TestPlanner(TestCase):
    def test_planner(self):
        random.seed(123)
        ctx = MagicMock()

        study = optuna.create_study(sampler=RandomSampler())
        trials = SmartTuningTrials(space=search_space(study=study))

        def get_trials():
            return trials

        ctx.get_current_config = get_current_config(trials, )
        ctx.get_from_engine = get_from_engine(trials)
        ctx.get_smarttuning_trials = get_trials

        tsampler = MagicMock()
        tsampler.metric = metric(production=False)
        psampler = MagicMock()
        psampler.metric = metric(production=True)

        t = Instance(
            name='daytrader-servicesmarttuning',
            namespace='default',
            is_production=False,
            sample_interval_in_secs=0,
            ctx=ctx,
            sampler=tsampler
        )
        t.restart = MagicMock(return_value=None)
        p = Instance(
            name='daytrader-services',
            namespace='default',
            is_production=False,
            sample_interval_in_secs=0,
            ctx=ctx,
            sampler=psampler
        )
        p.restart = MagicMock(return_value=None)

        planner = Planner(p, t, ctx, k=10, ratio=0.3334)

        def annonymous(reinforcement=None, best=None):
            print([cfg['name'] for cfg in best])

        planner.save_trace = annonymous
        planner.wait_for_metrics = wait_for_metrics(planner)

        results = []
        try:
            for _ in range(10):
                c: tuple[Configuration, bool] = copy.deepcopy(next(planner))
                results.append(f'{c[0].name}, {c[0].score:.2f}, {c[0].median():.2f}')
                print({k:v for k,v in optuna.importance.get_param_importances(study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()).items()})
        finally:
            from pprint import pprint
            pprint(results)

        print(planner.heap1)
        print(planner.heap2)

        print(optuna.importance.get_param_importances(study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()))

    def test_update_heap(self):
        c1 = Configuration(data={'name': 1}, trials=MagicMock())
        c2 = Configuration(data={'name': 2}, trials=MagicMock())
        a = Configuration(data={'name': 1}, trials=MagicMock())
        b = Configuration(data={'name': 2}, trials=MagicMock())
        c = Configuration(data={'name': 3}, trials=MagicMock())

        planner = Planner(MagicMock(), MagicMock(), MagicMock(), k=0, ratio=0)
        planner.heap1 = [c1]
        planner.heap2 = [c2]
        self.assertIs(planner.heap1[0], c1)
        self.assertIs(planner.heap2[0], c2)
        planner.update_heap(planner.heap1, a)
        self.assertIs(planner.heap1[0], a)

        planner.update_heap(planner.heap2, b)
        self.assertIs(planner.heap2[0], b)

        planner.update_heap(planner.heap1, c)
        self.assertIn(c, planner.heap1)
        self.assertIn(a, planner.heap1)

        planner.update_heap(planner.heap2, c)
        self.assertIn(c, planner.heap2)
        self.assertIn(b, planner.heap2)


if __name__ == '__main__':
    # random.seed(123)
    # SEED = 0
    # hashseed = os.getenv('PYTHONHASHSEED')
    # if not hashseed:
    #     os.environ['PYTHONHASHSEED'] = str(SEED)
    #     os.execv(sys.executable, [sys.executable] + sys.argv)
    unittest.main()
