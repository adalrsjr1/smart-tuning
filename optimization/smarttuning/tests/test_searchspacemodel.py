import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from optuna.samplers import RandomSampler

from controllers import searchspacemodel
from controllers.searchspacemodel import *
from mock_searchspace import *


class TestSearchModel(TestCase):

    def test_option_range_model(self):
        orm = OptionRangeModel({'name': 'x', 'type': 'integer', 'values': ['1', 2, 3.0]})
        trial_1 = optuna.trial.create_trial(
            params={orm.name: 1},
            distributions={
                orm.name: CategoricalDistribution(orm.get_values()),
            },
            value=0,
        )
        self.assertIsInstance(orm.sample(trial_1, {}), int)

        trial_2 = optuna.trial.create_trial(
            params={orm.name: 2},
            distributions={
                orm.name: CategoricalDistribution(orm.get_values()),
            },
            value=0,
        )
        self.assertEqual(2, orm.sample(trial_2, {}))
        self.assertIsInstance(orm.sample(trial_2, {}), int)

        trial_3 = optuna.trial.create_trial(
            params={orm.name: 3.0},
            distributions={
                orm.name: CategoricalDistribution(orm.get_values()),
            },
            value=0,
        )
        self.assertEqual(3, orm.sample(trial_3, {}))
        self.assertIsInstance(orm.sample(trial_3, {}), int)

    def test_number_range_model(self):
        nrm = NumberRangeModel({
            'name': 'x',
            'real': 'True',
            'lower': {'value': 0, 'dependsOn': ''},
            'upper': {'value': 10, 'dependsOn': ''},
            'step': None
        })
        trial_1 = optuna.trial.create_trial(
            params={nrm.name: 3.14},
            distributions={
                nrm.name: UniformDistribution(low=nrm.lower, high=nrm.upper),
            },
            value=0,
        )

        self.assertEqual(3.14, nrm.sample(trial_1, {}))
        self.assertIsInstance(nrm.sample(trial_1, {}), float)

    def test_number_range_model_with_dep(self):
        study = optuna.create_study(sampler=RandomSampler())
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

        def objective(trial: optuna.trial.Trial):
            sample = ctx.sample(trial)
            a = sample['a']
            b = sample['b']
            c = sample['c']
            x = sample['x']
            print(f'a:{a}, b:{b}, c:{c} x:{x}')
            self.assertTrue(40 <= a <= 55, msg=f'40 <= {a} <= 55')
            self.assertTrue(40 <= b <= 55, msg=f'40 <= {b} <= 55')
            self.assertTrue(40 <= c <= 55, msg=f'40 <= {c} <= 55')

            return 0

        study.optimize(objective, n_trials=1000)

    def test_real_search_space(self):
        study = optuna.create_study(sampler=RandomSampler())
        ctx = SearchSpaceModel(mock_daytrader_ss(), study)

        def objective(trial: optuna.trial.Trial):
            sample = ctx.sample(trial)
            max_threads = sample['MAX_THREADS']
            conmgr1_max_pool_size = sample['CONMGR1_MAX_POOL_SIZE']
            conmgr1_min_pool_size = sample['CONMGR1_MIN_POOL_SIZE']
            conmgr1_timeout = sample['CONMGR1_TIMEOUT']
            conmgr1_aged_timeout = sample['CONMGR1_AGED_TIMEOUT']
            conmgr1_max_idle_timeout = sample['CONMGR1_MAX_IDLE_TIMEOUT']
            conmgr1_reap_time = sample['CONMGR1_REAP_TIME']
            http_max_keep_alive_requests = sample['HTTP_MAX_KEEP_ALIVE_REQUESTS']
            http_persist_timeout = sample['HTTP_PERSIST_TIMEOUT']

            xms = sample['-Xms']
            xmx = sample['-Xmx']
            xmn = sample['-Xmn']
            shared_cache_hard_limit = sample['-XX:SharedCacheHardLimit']
            xscmx = sample['-Xscmx']
            cpu = sample['cpu']
            memory = sample['memory']
            virtualized = sample['-Xtune:virtualized']
            gcpolicy = sample['gc']
            container_support = sample['container_support']

            self.assertTrue(4 <= max_threads <= conmgr1_max_pool_size, f'max_threads [{trial.number}]: 4 <= {max_threads} <= {conmgr1_max_pool_size}')
            self.assertTrue(4 <= conmgr1_max_pool_size <= 100, f'conmgr1_max_pool_size [{trial.number}]: 4 <= {conmgr1_max_pool_size} <= 100')
            self.assertTrue(4 <= conmgr1_min_pool_size <= conmgr1_max_pool_size, f'conmgr1_min_pool_size [{trial.number}]: 4 <= {conmgr1_min_pool_size} <= {conmgr1_max_pool_size}')
            self.assertTrue(1 <= conmgr1_timeout <= 300, f'conmgr1_timeout [{trial.number}]: 1 <= {conmgr1_timeout} <= 300')
            self.assertTrue(1 <= conmgr1_aged_timeout <= 300, f'conmgr1_aged_timeout [{trial.number}]: 1 <= {conmgr1_aged_timeout} <= 300')
            self.assertTrue(1 <= conmgr1_max_idle_timeout <= 300, f'conmgr1_max_idle_timeout [{trial.number}]: 4 <= {conmgr1_max_idle_timeout} <= 300')
            self.assertTrue(1 <= conmgr1_reap_time <= conmgr1_max_idle_timeout, f'conmgr1_reap_time [{trial.number}]: 1 <= {conmgr1_reap_time} <= {conmgr1_max_idle_timeout}')
            self.assertTrue(max_threads <= http_max_keep_alive_requests <= conmgr1_max_pool_size, f'http_max_keep_alive_requests [{trial.number}]: {max_threads} <= {http_max_keep_alive_requests} <= {conmgr1_max_pool_size}')
            self.assertTrue(15 <= http_persist_timeout <= 45, f'http_persist_timeout [{trial.number}]: 15 <= {http_persist_timeout} <= {45}')

            self.assertIn(virtualized, [True, False], f'virtualized [{trial.number}]')

            self.assertTrue(8 <= xms <= 896, f'xms [{trial.number}]: 8 <= {xms} <= {896}')
            self.assertLessEqual(trial.distributions["-Xmn"].low, xms)
            self.assertTrue(trial.distributions["-Xmx"].low <= xmx <= memory * 0.8, f'xmx [{trial.number}]: {trial.distributions["-Xmx"].low} <= {xmx} <= {memory * 0.8}')
            self.assertLessEqual(trial.distributions["-Xmn"].low, 8)
            self.assertTrue(trial.distributions["-Xmn"].low <= xmn <= int(xms * 0.8), f'xmn [{trial.number}]: {trial.distributions["-Xmn"].low} <= {xmn} <= {int(xms*0.8)}')
            self.assertTrue(16 <= xscmx <= 512, f'xscmx [{trial.number}]: 16 <= {xscmx} <= {512}')
            self.assertTrue(16 <= shared_cache_hard_limit <= 512, f'shared_cache_hard_limit [{trial.number}]: 16 <= {shared_cache_hard_limit} <= {512}')

            self.assertIn(gcpolicy, ['-Xgcpolicy:gencon'], f'gcpolicy [{trial.number}]')
            self.assertIn(container_support, ['-XX:+UseContainerSupport'], f'container_support [{trial.number}]')

            self.assertIn(memory, [1024, 2048, 4096, 8192], f'memory [{trial.number}]')
            self.assertIn(cpu, [2, 4, 6], f'cpu [{trial.number}]')


            return 0

        study.optimize(objective, n_trials=1000)

    def test_sample(self):
        study = optuna.create_study(sampler=RandomSampler())
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

        trial = optuna.trial.create_trial(
            params={"a": 49, "b": 49, "c": 49, "x": 53},
            distributions={
                "a": UniformDistribution(low=0, high=100),
                "b": UniformDistribution(low=25, high=75),
                "c": UniformDistribution(low=40, high=50),
                "x": CategoricalDistribution([52, 53, 55]),
            },
            value=0,
        )

        print(ctx.sample(trial, full=True))

    def test_number_range_model_with_mathemathical_dep(self):
        study = optuna.create_study(sampler=RandomSampler())
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
                                    "dependsOn": "x 0.81 *",
                                    "value": 50
                                }
                            },
                        ]
                    }
                },
            ],
        }, study)

        def objective(trial: optuna.trial.Trial):
            sample = ctx.sample(trial)
            a = sample['a']
            b = sample['b']
            c = sample['c']
            x = sample['x']
            print(f'a:{a}, b:{b}, c:{c} x:{x}')
            self.assertTrue(40 <= a <= 55 * 0.81, msg=f'40 <= {a} <= {55 * 0.81}')
            self.assertTrue(40 <= b <= 55 * 0.81, msg=f'40 <= {b} <= {55 * 0.81}')
            self.assertTrue(40 <= c <= 55 * 0.81, msg=f'40 <= {c} <= {55 * 0.81}')

            return 0

        study.optimize(objective, n_trials=1000)

    @classmethod
    def setUpClass(cls) -> None:
        cls.ss = mock_acmeair_search_space()
        cls.ss_dep_continuous = mock_acmeair_search_space_dep_continous()

    def test_jmvoptions_to_dict(self):
        jvm_options_file = "#-Dcom.sun.management.jmxremote\n#-Dcom.sun.management.jmxremote.authenticate=false\n#-Dcom.sun.management.jmxremote.ssl=false\n#-Dcom.sun.management.jmxremote.local.only=false\n#-Dcom.sun.management.jmxremote.port=1099\n#-Dcom.sun.management.jmxremote.rmi.port=1099\n#-Djava.rmi.server.hostname=127.0.0.1\n-XX:+UseContainerSupport\n#-xms512m\n#-xmx512m\n-Xgcpolicy:gencon\n-Xtune:virtualized\n-XX:InitialRAMPercentage=25\n-XX:MaxRAMPercentage=75\n-Xmn128m\n-XX:SharedCacheHardLimit=32m\n-Xscmx=16m\n#-verbose:gc\n#-Xverbosegclog:/home/daytrader/verbosegc.%Y%m%d.%H%M%S.%pid.txt"
        _, params = jmvoptions_to_dict(jvm_options_file)
        from pprint import pprint
        pprint(params)
        self.assertDictEqual(params, {
            '-XX:InitialRAMPercentage': 25,
            '-XX:MaxRAMPercentage': 75,
            '-Xmn': 128,
            '-XX:SharedCacheHardLimit': 32,
            '-Xscmx': 16,
            '-Xtune:virtualized': True,
            'container_support': '-XX:+UseContainerSupport',
            'gc': '-Xgcpolicy:gencon',

        })

    def test_searchspace_model(self):
        study = optuna.create_study(sampler=RandomSampler())

        ss = SearchSpaceModel(self.ss, study)

        self.assertEqual(ss.deployment, "acmeair-service")
        self.assertEqual(len(ss.manifests), 3)

        self.assertEqual(3, len(ss.manifests))

    def test_searchspace_deployment(self):
        study = optuna.create_study(sampler=RandomSampler())
        ss = SearchSpaceModel(self.ss, study)

        ss_dep = [manifest for manifest in ss.manifests if isinstance(manifest, DeploymentSearchSpaceModel)][0]
        self.assertSetEqual({'cpu', 'memory'}, set(ss_dep.tunables.keys()))

    def test_searchspace_deployment_get_current_config(self):
        study = optuna.create_study(sampler=RandomSampler())
        ss = SearchSpaceModel(self.ss, study)

        ss_dep = [manifest for manifest in ss.manifests if isinstance(manifest, DeploymentSearchSpaceModel)][0]
        self.assertSetEqual(set(ss_dep.tunables.keys()),
                            set(ss_dep.get_current_config_from_spec(mock_acmeair_deployment())))

        # per container
        # {'cpu': [2, 4, 8]}
        # {'memory': ['512', '1024', '2048', '4096', '8192']}
        # {'limits':{'cpu': '2', 'memory':'4096'}
        indexed, valued = ss_dep.get_current_config_from_spec(mock_acmeair_deployment())
        self.assertDictEqual({'cpu': 1, 'memory': 4}, indexed)
        self.assertDictEqual({'cpu': 4, 'memory': 8192}, valued)

    def test_searchspace_deployment_get_current_config(self):
        study = optuna.create_study(sampler=RandomSampler())
        ss = SearchSpaceModel(self.ss_dep_continuous, study)

        ss_dep = [manifest for manifest in ss.manifests if isinstance(manifest, DeploymentSearchSpaceModel)][0]
        self.assertSetEqual(set(ss_dep.tunables.keys()),
                            set(ss_dep.get_current_config_from_spec(mock_acmeair_deployment())))

        # per container
        # {'cpu': [2, 4, 8]}
        # {'memory': ['512', '1024', '2048', '4096', '8192']}
        # {'limits':{'cpu': '2', 'memory':'4096'}
        valued = ss_dep.get_current_config_from_spec(mock_acmeair_deployment())
        self.assertDictEqual({'cpu': 4, 'memory': 8192}, valued)

    def test_patch_deployment(self):
        study = optuna.create_study(sampler=RandomSampler())
        ss = SearchSpaceModel(self.ss_dep_continuous, study)
        ss_dep = [manifest for manifest in ss.manifests if isinstance(manifest, DeploymentSearchSpaceModel)][0]

        searchspacemodel.get_deployment = mock_acmeair_deployment

        # 2 containers
        nane, namespace, containers = ss_dep.core_patch({'cpu': 2, 'memory': 512}, production=True)
        container: V1Container
        for container in containers:
            self.assertDictEqual({'cpu': '1.0', 'memory': f'{256}Mi'}, container.resources.limits)

    def test_configs_maps(self):
        study = optuna.create_study(sampler=RandomSampler())
        ss = SearchSpaceModel(self.ss_dep_continuous, study)
        ss_cms = [manifest for manifest in ss.manifests if isinstance(manifest, ConfigMapSearhSpaceModel)]

        self.assertEqual(2, len(ss_cms))

    def test_get_current_config_map_app(self):
        study = optuna.create_study(sampler=RandomSampler())
        ss = SearchSpaceModel(self.ss, study)
        ss_cm = [manifest for manifest in ss.manifests if
                 isinstance(manifest, ConfigMapSearhSpaceModel) and 'acmeair-config-app' == manifest.name][0]

        valued = ss_cm.get_current_config_core(mock_acmeair_app_cm())
        # {
        #   "EXECUTOR_STEAL_POLICY": ["STRICT", "LOCAL", "NEVER"],
        #   "HTTP_MAX_KEEP_ALIVE_REQUESTS": (10, 200, 25),
        #   "HTTP_PERSIST_TIMEOUT": (15, 45, 5),
        #   "MONGO_MAX_CONNECTIONS": (10, 200, 25),
        # }

        print(valued)
        self.assertDictEqual({
            "EXECUTOR_STEAL_POLICY": "LOCAL",
            "HTTP_MAX_KEEP_ALIVE_REQUESTS": "100",
            "HTTP_PERSIST_TIMEOUT": "30",
            "MONGO_MAX_CONNECTIONS": "100",
        }, valued)

    def test_get_current_config_map_jvm_eval_with_search_space(self):
        study = optuna.create_study(sampler=RandomSampler())
        ss = SearchSpaceModel(self.ss, study)
        ss_cm = [manifest for manifest in ss.manifests if
                 isinstance(manifest, ConfigMapSearhSpaceModel) and 'acmeair-config-jvm' == manifest.name][0]

        self.assertDictEqual({
            'gc': '-Xgcpolicy:gencon',
            'container_support': '-XX:+UseContainerSupport',
            '-Xtune:virtualized': False
        }, ss_cm.get_current_config_core(mock_acmeair_jvm_cm()))

    def test_Option_range(self):
        r = {'name': 'test_name', 'type': 'bool', 'values': ['True', 'False']}
        orm = OptionRangeModel(r)
        self.assertListEqual([True, False], orm.get_values())

        r = {'name': 'test_name', 'type': 'integer', 'values': [1, 2, 3]}
        orm = OptionRangeModel(r)
        self.assertListEqual([1, 2, 3], orm.get_values())

        r = {'name': 'test_name', 'type': 'real', 'values': [3.14, 2.71, 1.618]}
        orm = OptionRangeModel(r)
        self.assertListEqual([3.14, 2.71, 1.618], orm.get_values())

    def test_dep_regex(self):
        trial = optuna.trial.create_trial(
            params={"name": 1},
            distributions={
                "name": UniformDistribution(0, 2),
            },
            value=0,
        )
        ctx = MagicMock()

        def tunables():
            return {'name': NumberRangeModel({
                'name': 'name',
                'real': True,
                'upper': {'value': 2, 'dependsOn': ''},
                'lower': {'value': 0, 'dependsOn': ''}
            }, ctx)}

        ctx.tunables = tunables

        self.assertEqual(dep_eval('0.95 name * ', ctx, trial, {}), 0.95)
        self.assertEqual(dep_eval('name 0.95 *', ctx, trial, {}), 0.95)
        self.assertEqual(dep_eval('name 1 *', ctx, trial, {}), 1.0)
        self.assertEqual(dep_eval('name 1.000 *', ctx, trial, {}), 1.0)
        self.assertEqual(dep_eval('name 2 *', ctx, trial, {}), 2.0)
        self.assertEqual(dep_eval('1 name +', ctx, trial, {}), 2.0)
        self.assertEqual(dep_eval('1.000 name + ', ctx, trial, {}), 2.0)
        self.assertEqual(dep_eval('2 name * ', ctx, trial, {}), 2)
        self.assertEqual(dep_eval('name 1 * ', ctx, trial, {}), 1)
        self.assertEqual(dep_eval('1 name * ', ctx, trial, {}), 1)
        self.assertEqual(dep_eval('name 1.00 * ', ctx, trial, {}), 1)
        self.assertEqual(dep_eval('1.1234 name * ', ctx, trial, {}), 1.1234)
        with self.assertRaises(TypeError):
            dep_eval('test', ctx, trial, {})

    def test_dep_eval(self):
        trial = optuna.trial.create_trial(
            params={"VAR": 7},
            distributions={
                "VAR": UniformDistribution(0, 10),
            },
            value=0,
        )
        ctx = MagicMock()

        def tunables():
            return {'VAR': NumberRangeModel({
                'name': 'VAR',
                'real': True,
                'upper': {'value': 10, 'dependsOn': ''},
                'lower': {'value': 0, 'dependsOn': ''}
            }, ctx)}

        ctx.tunables = tunables

        with self.assertRaises(TypeError):
            eval_token('VAT', ctx, trial)
        self.assertEqual(eval_token('1', ctx, trial, {}), 1.0)
        self.assertEqual(eval_token('1.1', ctx, trial, {}), 1.1)
        self.assertEqual(eval_token('+', ctx, trial, {}), '+')
        self.assertEqual(eval_token('VAR', ctx, trial, {}), 7)

    def test_tokenize(self):
        trial = optuna.trial.create_trial(
            params={"NAME": 3.14},
            distributions={
                "NAME": UniformDistribution(0, 5),
            },
            value=0,
        )
        ctx = MagicMock()

        def tunables():
            return {'NAME': NumberRangeModel({
                'name': 'NAME',
                'real': True,
                'upper': {'value': 5, 'dependsOn': ''},
                'lower': {'value': 0, 'dependsOn': ''}
            }, ctx)}

        ctx.tunables = tunables
        expr = 'NAME 3 +'
        self.assertEqual([3.14, 3, '+'], tokenize(expr, ctx, trial, {}))
        with self.assertRaises(TypeError):
            tokenize('', ctx, trial)

    def test_polish_eval(self):
        self.assertEqual(-17, polish_eval([3, 4, 5, '*', '-']))
        self.assertEqual(14, polish_eval([5, 1, 2, '+', 4, '*', '+', 3, '-']))
        self.assertEqual(35, polish_eval([3, 4, '+', 5, '*']))
        self.assertEqual(15, polish_eval([3, 4, '+', 2, '*', 1, '+']))
        self.assertEqual(0, polish_eval([]))
        from hyperopt import hp
        a = hp.uniform('a', 0, 1)
        self.assertEqual(a, polish_eval([a]))
        self.assertEqual(None, polish_eval(['-']))

    def test_searchspace_model(self):
        study = optuna.create_study(sampler=RandomSampler(seed=0))
        raw_object = mock_daytrader_ss()
        model = SearchSpaceModel(raw_object, study)

        for _ in range(100):
            trial = model.adhoc_trial()
            print(model.sample(trial, full=True))

    def test_adhoc_trial(self):
        study = optuna.create_study(sampler=RandomSampler(seed=0))
        raw_object = mock_daytrader_ss()
        model = SearchSpaceModel(raw_object, study)

        t = model.adhoc_trial()



if __name__ == '__main__':
    unittest.main()
