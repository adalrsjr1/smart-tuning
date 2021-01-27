import unittest
from unittest import TestCase, main
import json
from controllers.searchspacemodel import *
from controllers import searchspacemodel
from mock_searchspace import *


class TestSearchModel(TestCase):

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
            '-Xmn':128,
            '-XX:SharedCacheHardLimit': 32,
            '-Xscmx': 16,
            '-Xtune:virtualized': True,
            'container_support': '-XX:+UseContainerSupport',
            'gc': '-Xgcpolicy:gencon',

        })

    def test_searchspace_model(self):
        ss = SearchSpaceModel(self.ss)

        self.assertEqual(ss.deployment, "acmeair-service")
        self.assertEqual(len(ss.manifests), 3)

        self.assertEqual(3, len(ss.manifests))

    def test_searchspace_deployment(self):
        ss = SearchSpaceModel(self.ss)

        ss_dep = [manifest for manifest in ss.manifests if isinstance(manifest, DeploymentSearchSpaceModel)][0]
        self.assertSetEqual({'cpu', 'memory'}, set(ss_dep.search_space().keys()))

    def test_searchspace_deployment_get_current_config(self):
        ss = SearchSpaceModel(self.ss)

        ss_dep = [manifest for manifest in ss.manifests if isinstance(manifest, DeploymentSearchSpaceModel)][0]
        self.assertSetEqual(set(ss_dep.search_space().keys()),
                            set(ss_dep.get_current_config_from_spec(mock_acmeair_deployment())[0]))
        self.assertSetEqual(set(ss_dep.search_space().keys()),
                            set(ss_dep.get_current_config_from_spec(mock_acmeair_deployment())[1]))

        # per container
        # {'cpu': [2, 4, 8]}
        # {'memory': ['512', '1024', '2048', '4096', '8192']}
        # {'limits':{'cpu': '2', 'memory':'4096'}
        indexed, valued = ss_dep.get_current_config_from_spec(mock_acmeair_deployment())
        self.assertDictEqual({'cpu': 1, 'memory': 4}, indexed)
        self.assertDictEqual({'cpu': 4, 'memory': 8192}, valued)

    def test_searchspace_deployment_get_current_config(self):
        ss = SearchSpaceModel(self.ss_dep_continuous)

        ss_dep = [manifest for manifest in ss.manifests if isinstance(manifest, DeploymentSearchSpaceModel)][0]
        self.assertSetEqual(set(ss_dep.search_space().keys()),
                            set(ss_dep.get_current_config_from_spec(mock_acmeair_deployment())[0]))
        self.assertSetEqual(set(ss_dep.search_space().keys()),
                            set(ss_dep.get_current_config_from_spec(mock_acmeair_deployment())[1]))

        # per container
        # {'cpu': [2, 4, 8]}
        # {'memory': ['512', '1024', '2048', '4096', '8192']}
        # {'limits':{'cpu': '2', 'memory':'4096'}
        indexed, valued = ss_dep.get_current_config_from_spec(mock_acmeair_deployment())
        self.assertDictEqual({'cpu': 4, 'memory': 8192}, valued)

    def test_patch_deployment(self):
        ss = SearchSpaceModel(self.ss)
        ss_dep = [manifest for manifest in ss.manifests if isinstance(manifest, DeploymentSearchSpaceModel)][0]

        searchspacemodel.get_deployment = mock_acmeair_deployment

        # 2 containers
        nane, namespace, containers = ss_dep.core_patch({'cpu': 2, 'memory': 512}, production=True)
        container: V1Container
        for container in containers:
            self.assertDictEqual({'cpu': '1.0', 'memory': f'{256}Mi'}, container.resources.limits)

    def test_configs_maps(self):
        ss = SearchSpaceModel(self.ss)
        ss_cms = [manifest for manifest in ss.manifests if isinstance(manifest, ConfigMapSearhSpaceModel)]

        self.assertEqual(2, len(ss_cms))

    def test_get_current_config_map_app(self):
        ss = SearchSpaceModel(self.ss)
        ss_cm = [manifest for manifest in ss.manifests if
                 isinstance(manifest, ConfigMapSearhSpaceModel) and 'acmeair-config-app' == manifest.name][0]

        indexed, valued = ss_cm.get_current_config_core(mock_acmeair_app_cm())
        # {
        #   "EXECUTOR_STEAL_POLICY": ["STRICT", "LOCAL", "NEVER"],
        #   "HTTP_MAX_KEEP_ALIVE_REQUESTS": (10, 200, 25),
        #   "HTTP_PERSIST_TIMEOUT": (15, 45, 5),
        #   "MONGO_MAX_CONNECTIONS": (10, 200, 25),
        # }

        self.assertDictEqual({
            "EXECUTOR_STEAL_POLICY": 1,
            "HTTP_MAX_KEEP_ALIVE_REQUESTS": "100",
            "HTTP_PERSIST_TIMEOUT": "30",
            "MONGO_MAX_CONNECTIONS": "100",
        }, indexed)
        self.assertDictEqual({
            "EXECUTOR_STEAL_POLICY": "LOCAL",
            "HTTP_MAX_KEEP_ALIVE_REQUESTS": "100",
            "HTTP_PERSIST_TIMEOUT": "30",
            "MONGO_MAX_CONNECTIONS": "100",
        }, valued)

    def test_get_current_config_map_jvm_eval_with_search_space(self):
        ss = SearchSpaceModel(self.ss)
        ss_cm = [manifest for manifest in ss.manifests if
                 isinstance(manifest, ConfigMapSearhSpaceModel) and 'acmeair-config-jvm' == manifest.name][0]

        jvm_ss = ss.search_space()['acmeair-config-jvm']
        from hyperopt import space_eval
        self.assertDictEqual({
            'gc': '-Xgcpolicy:gencon',
            'container_support': '-XX:+UseContainerSupport',
            '-Xtune:virtualized': False
        }, space_eval(jvm_ss, ss_cm.get_current_config_core(mock_acmeair_jvm_cm())[0]))


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

    def test_number_range(self):

        r1 = {
            'dependsOn': '',
            'name': 'test_name',
            'upper': {
                'value': 100,
                'dependsOn':'another_name'
            },
            'lower': {
                'value': 3,
                'dependsOn':''
            },
            'step': 1,
            'real': True
        }
        r2 = {
            'dependsOn': '',
            'name': 'another_name',
            'upper': {
                'value': 10,
                'dependsOn':''
            },
            'lower': {
                'value': 5,
                'dependsOn':''
            },
            'step': 1,
            'real': True
        }
        nrm1 = NumberRangeModel(r1)
        nrm2 = NumberRangeModel(r2)
        tunables = {'test_name':nrm1, 'another_name':nrm2}
        value = nrm1.get_hyper_interval(ctx=tunables)
        import numpy as np
        from hyperopt.pyll.stochastic import sample
        count = 0
        for i in range(1000):
            x = sample(value, np.random.RandomState(i))['test_name']
            if x == 0:
                count += 1
            self.assertTrue(3 <= x <= 10, msg=f'{x}')
            self.assertNotEqual(1000, count)

    def test_number_range_expr(self):

        r1 = {
            'dependsOn': '',
            'name': 'test_name',
            'upper': {
                'value': 100,
                'dependsOn':'another_name 0.95 *'
            },
            'lower': {
                'value': 50,
                'dependsOn':''
            },
            'step': 1,
            'real': True
        }
        r2 = {
            'dependsOn': '',
            'name': 'another_name',
            'upper': {
                'value': 100,
                'dependsOn':''
            },
            'lower': {
                'value': 10,
                'dependsOn':''
            },
            'step': 1,
            'real': True
        }
        nrm1 = NumberRangeModel(r1)
        nrm2 = NumberRangeModel(r2)
        tunables = {'test_name':nrm1, 'another_name':nrm2}
        value = nrm1.get_hyper_interval(ctx=tunables)
        import numpy as np
        from hyperopt.pyll.stochastic import sample
        count = 0
        for i in range(1000):
            x = sample(value, np.random.RandomState(i))['test_name']
            if x == 0:
                count += 1
            self.assertTrue(10*0.95 <= x <= 100*0.95, msg=f'{i}:{x}')
            self.assertNotEqual(1000, count)

    def test_number_option_range(self):

        r1 = {
            'dependsOn': '',
            'name': 'test_name',
            'upper': {
                'value': 100,
                'dependsOn':'another_name 0.95 *'
            },
            'lower': {
                'value': 20,
                'dependsOn':''
            },
            'step': 1,
            'real': True
        }
        r2 = {
            'name': 'another_name',
            'type': 'integer',
            'values': [20, 30, 40]
        }
        nrm1 = NumberRangeModel(r1)
        nrm2 = OptionRangeModel(r2)
        tunables = {'test_name':nrm1, 'another_name':nrm2}
        value = nrm1.get_hyper_interval(ctx=tunables)
        import numpy as np
        from hyperopt.pyll.stochastic import sample
        count = 0
        for i in range(1000):
            x = sample(value, np.random.RandomState(i))['test_name']
            if x == 0:
                count += 1
            self.assertTrue(20*0.95 <= x <= 40*0.95, msg=f'{i}:{x}')
            self.assertNotEqual(1000, count)

    def test_to_scale(self):
        from hyperopt import hp
        a = hp.uniform('a', 0, 8192) * 0.95
        print('...', to_scale(8, 8, a, 1024, 7680))

    def test_dep_regex(self):
        self.assertEqual(dep_eval('0.95 name * ', {'name': 1.0}), 0.95)
        self.assertEqual(dep_eval('name 0.95 *', {'name': 1.0}), 0.95)
        self.assertEqual(dep_eval('name 1 *', {'name': 1.0}),1.0)
        self.assertEqual(dep_eval('name 1.000 *', {'name': 1.0}), 1.0)
        self.assertEqual(dep_eval('name 2 *', {'name': 1.0}), 2.0)
        self.assertEqual(dep_eval('1 name +', {'name': 1.0}), 2.0)
        self.assertEqual(dep_eval('1.000 name + ', {'name': 1.0}), 2.0)
        self.assertEqual(dep_eval('2 name * ', {'name': 1.0}), 2)
        self.assertEqual(dep_eval('name 1 * ', {'name': 1.0}), 1)
        self.assertEqual(dep_eval('1 name * ', {'name': 1.0}), 1)
        self.assertEqual(dep_eval('name 1.00 * ', {'name': 1.0}), 1)
        self.assertEqual(dep_eval('1.1234 name * ', {'name': 1.0}), 1.1234)
        import math
        with self.assertRaises(KeyError):
            dep_eval('test', {'name': 1.0})
        self.assertTrue(math.isnan(dep_eval('test 1 + ', {'name': 1.0}, default=float('nan'))))
        self.assertEqual(dep_eval('test 1 + ', {'name': 1.0}, default=3), 4)

    def test_dep_eval(self):
        self.assertEqual(dep_eval_expr('VAT', {'VAR': 7}, default={'VAT':5}), 5)
        self.assertEqual(dep_eval_expr('VAT', {'VAR': 7}, default=3.14), 3.14)
        with self.assertRaises(KeyError):
            dep_eval_expr('VAT', {'VAR': 7})
        self.assertEqual(dep_eval_expr('1'), 1.0)
        self.assertEqual(dep_eval_expr('1.1'), 1.1)
        self.assertEqual(dep_eval_expr('+'), '+')
        self.assertEqual(dep_eval_expr('VAR', {'VAR': 7}), 7)



    def test_tokenize(self):
        expr = 'NAME 3 +'
        self.assertEqual([3.14, 3, '+'], tokenize(expr, {'NAME':3.14}))
        self.assertEqual([7], tokenize('', {'NAME':3.14}, default=7))

    def test_polish_eval(self):

        self.assertEqual(-17,polish_eval([3, 4, 5, '*', '-']))
        self.assertEqual(14, polish_eval([5,1,2,'+',4,'*','+',3,'-']))
        self.assertEqual(35, polish_eval([3, 4, '+', 5, '*']))
        self.assertEqual(15, polish_eval([3, 4, '+', 2, '*', 1, '+']))
        self.assertEqual(0,polish_eval([]))
        from hyperopt import hp
        a = hp.uniform('a', 0, 1)
        self.assertEqual(a, polish_eval([a]))
        self.assertEqual(None, polish_eval(['-']))







if __name__ == '__main__':
    unittest.main()
