import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import optuna
from optuna.samplers import TPESampler

from bayesian import BayesianEngine, BayesianDTO
from controllers.searchspacemodel import SearchSpaceModel
from models.configuration import Configuration
from sampler import Metric


class TestBayesianEngine(unittest.TestCase):

    def test_e2e_bayesian(self):
        max_evals = 10
        study = optuna.create_study(sampler=TPESampler())
        space = SearchSpaceModel({
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
                        "option": [
                            {
                                "name": "x",
                                "type": "integer",
                                "values": [
                                    "52",
                                    "53",
                                    "55"
                                ]
                            },
                        ],
                        "number": [
                            {
                                "name": "b",
                                "real": True,
                                "lower": {
                                    "dependsOn": "c",
                                    "value": 25
                                },
                                "upper": {
                                    "dependsOn": "c",
                                    "value": 75
                                }
                            },
                            {
                                "name": "a",
                                "real": True,
                                "lower": {
                                    "dependsOn": "b",
                                    "value": 0
                                },
                                "upper": {
                                    "dependsOn": "b",
                                    "value": 100
                                }
                            },
                            {
                                "name": "c",
                                "real": True,
                                "lower": {
                                    "dependsOn": "",
                                    "value": 40
                                },
                                "upper": {
                                    "dependsOn": "x",
                                    "value": 50
                                }
                            },
                        ]
                    }
                },
            ],
        }, study)
        engine = BayesianEngine(
            name='test',
            space=space,
            max_evals=max_evals,
        )
        thread = engine.fmin

        def put():
            for i in range(max_evals):
                metric = Metric(to_eval=str(i))
                engine.put(BayesianDTO(metric=metric))

        def get():
            for i in range(1, max_evals + 1):
                c: Configuration = engine.get()
                print(c)

            self.assertEqual(i, max_evals)

        threading.Thread(name='put-metric', target=put).start()
        threading.Thread(name='get-metric', target=get).start()

        thread.join()

    executor = ThreadPoolExecutor()


if __name__ == '__main__':
    unittest.main()
