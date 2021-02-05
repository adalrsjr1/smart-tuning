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


class TestConfiguration(TestCase):
    def test_instantiate(self):
        study = study_filled(n=10)
        trials = SmartTuningTrials(space=search_space(study))
        trial = trials.last_trial()
        c = Configuration(trial=trial, trials=trials)

        self.assertDictEqual(c.data, trial.params)
        self.assertEqual(c.uid, trial.number)
        self.assertIs(c.trial, trial)

    def test_update_score(self):
        study = study_filled(n=10)
        trials = SmartTuningTrials(space=search_space(study))
        trial = trials.last_trial()
        c = Configuration(trial=trial, trials=trials)

        metric = Metric(to_eval='100')
        c.update_score(value=metric)
        self.assertEqual(c.score, trial.value)
        self.assertEqual(c.score, 100)
        self.assertEqual(trial.value, 100)

        print(c.serialize())


if __name__ == '__main__':
    unittest.main()
