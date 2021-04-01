import time
from unittest.mock import MagicMock

import optuna
from optuna import Trial
from optuna.distributions import CategoricalDistribution, UniformDistribution, IntUniformDistribution
from optuna.trial import BaseTrial

from controllers.searchspacemodel import SearchSpaceModel


class MockSearchSpace:
    @staticmethod
    def new():
        ss = SearchSpaceModel({
            "spec": {
                "deployment": "manifest-a",
                "manifests": [
                    {
                        "name": "manifest-a",
                        "type": "deployment"
                    },
                    {
                        "name": "manifest-b",
                        "type": "configMap"
                    },
                    {
                        "name": "manifest-c",
                        "type": "configMap"
                    },
                ],
                "namespace": "default",
                "service": "fake-svc"
            },
            "data": [
                {
                    "filename": "",
                    "name": "manifest-a",
                    "tunables": {
                        "number": [
                            {
                                "name": "a1",
                                "real": False,
                                "lower": {
                                    "dependsOn": "",
                                    "value": 0
                                },
                                "upper": {
                                    "dependsOn": "",
                                    "value": 10
                                }
                            },
                            {
                                "name": "a2",
                                "real": True,
                                "lower": {
                                    "dependsOn": "",
                                    "value": 5
                                },
                                "upper": {
                                    "dependsOn": "",
                                    "value": 6
                                }
                            },
                        ]
                    }
                },
                {
                    "filename": "",
                    "name": "manifest-b",
                    "tunables": {
                        "option": [
                            {
                                "name": "b1",
                                "type": "string",
                                "values": [
                                    "b1_x",
                                    "b2_y",
                                ]
                            },
                        ]
                    }
                },
                {
                    "filename": "",
                    "name": "manifest-c",
                    "tunables": {
                        "boolean": [
                            {
                                "name": "c1",
                            },
                        ],
                        "number": [
                            {
                                "name": "c2",
                                "real": False,
                                "lower": {
                                    "dependsOn": "",
                                    "value": 10
                                },
                                "upper": {
                                    "dependsOn": "",
                                    "value": 100
                                }
                            },
                        ]
                    }
                },
            ],
            # }, study)
        })
        for manifest in ss.manifests:
            manifest.get_current_config

            if manifest.name == 'manifest-a':
                manifest.get_current_config = MagicMock(return_value={
                    'a1': 5,
                    'a2': 5.5,
                })

            elif manifest.name == 'manifest-b':
                manifest.get_current_config = MagicMock(return_value={
                    'b1': 'b2_y',
                })
            elif manifest.name == 'manifest-c':
                manifest.get_current_config = MagicMock(return_value={
                    'c1': True,
                    'c2': 55,
                })
        return ss


class MockTrial:
    @staticmethod
    def create_trial(value: float = 3.14, frozen=True) -> BaseTrial:
        if not frozen:
            s = optuna.create_study(study_name=f'{id}')
            t = s.ask()
            s.tell(t, value)
            return t
        return optuna.trial.create_trial(
            params={
                'a1': 5,
                'a2': 5.5,
                'b1': 'b2_y',
                'c1': True,
                'c2': 55,
            },
            distributions={
                'a1': IntUniformDistribution(0, 10),
                'a2': UniformDistribution(5, 6),
                'b1': CategoricalDistribution(['b1_x', 'b2_y']),
                'c1': CategoricalDistribution([True, False]),
                'c2': IntUniformDistribution(10, 100)
            },
            value=value,
        )
