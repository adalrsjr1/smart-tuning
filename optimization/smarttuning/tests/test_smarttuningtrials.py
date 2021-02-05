import unittest
from unittest import TestCase

import optuna
from optuna.distributions import UniformDistribution
from optuna.samplers import RandomSampler

from controllers.searchspacemodel import SearchSpaceModel
from models.configuration import Configuration
from models.smartttuningtrials import SmartTuningTrials
from sampler import Metric


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
                            "name": "x",
                            "real": True,
                            "lower": {
                                "dependsOn": "",
                                "value": 0
                            },
                            "upper": {
                                "dependsOn": "",
                                "value": 100
                            }
                        }
                    ]
                }
            },
        ],
    }, study)
    return ctx


def study_filled(n=10):
    study = optuna.create_study(sampler=RandomSampler())

    for i in range(n):
        t = optuna.trial.create_trial(
            params={'x': i},
            distributions={'x': UniformDistribution(low=0, high=n + 1)},
            value=i * i
        )
        study.add_trial(t)

    return study


class TestSmartTuningTrials(TestCase):

    def test_instantiate(self):
        study = study_filled(n=10)
        space = search_space(study)
        t = SmartTuningTrials(space)

        self.assertIsNotNone(t)

    def test_serialize(self):
        n = 10
        study = study_filled(n=n)
        space = search_space(study)
        t = SmartTuningTrials(space)

        self.assertListEqual([{
            'tid': i,
            'params': {'x': i},
            'loss': i * i,
            'status': 'COMPLETE',
        } for i in range(n)], t.serialize())

    def test_last_uid(self):
        study = study_filled(n=10)
        space = search_space(study)
        t = SmartTuningTrials(space)

        self.assertEqual(9, t.last_uid())

    def test_add_default_config(self):
        n = 10
        study = study_filled(n=n)
        space = search_space(study)
        t = SmartTuningTrials(space)

        self.assertEqual(n - 1, t.last_uid())
        c = t.add_default_config(data={'x': 50}, metric=Metric(to_eval='50*50'))
        self.assertEqual(50 * 50, c.score)
        self.assertEqual(50 * 50, c.trial.value)
        self.assertEqual(n, t.last_uid(), msg=f'n:{n}, last_uid:{t.last_uid()}')

        self.assertEqual(10, t.last_uid())
        trial = t.wrapped_trials[-1]
        self.assertEqual(50 * 50, trial.value)
        self.assertEqual(50, trial.params['x'])

    def test_update_trial_score(self):
        n = 10
        study = study_filled(n=n)
        space = search_space(study)
        t = SmartTuningTrials(space)

        self.assertEqual(n - 1, t.last_uid())
        c = t.add_default_config(data={'x': 50}, metric=Metric(to_eval='50*50'))
        self.assertEqual(50 * 50, c.score)
        self.assertEqual(50 * 50, c.trial.value)
        self.assertEqual(n, t.last_uid(), msg=f'n:{n}, last_uid:{t.last_uid()}')

        c.update_score(Metric.zero())
        self.assertEqual(c.median(), c.trial.value)

    def test_default_config(self):
        n = 10
        study = study_filled(n=n)
        space = search_space(study)
        t = SmartTuningTrials(space)

        t.add_default_config({'a': {'x': 0}}, Metric.zero())

if __name__ == '__main__':
    unittest.main()
