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


if __name__ == '__main__':
    unittest.main()
