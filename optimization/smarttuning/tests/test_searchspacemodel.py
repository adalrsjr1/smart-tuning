from unittest import TestCase, main
import json
from controllers.searchspacemodel import *

class TestSearchModel(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config.init_k8s(hostname='trxrhel7perf-1')
        with open('search_space.json') as json_file:
            cls.ss = json.load(json_file)['object']

    def test_searchspace_model(self):
        ss = SearchSpaceModel(self.ss)

        self.assertEqual(ss.deployment, "acmeair-nginx-test-service")
        self.assertEqual(len(ss.manifests), 3)

        for manifest in ss.manifests:
            print(manifest.search_space())

    def test_patch_deployment(self):
        ss = SearchSpaceModel(self.ss)

        containers = [{'name': 'nginx-test'}]
        for manifest in ss.manifests:
            if isinstance(manifest, DeploymentSearchSpaceModel):
                manifest.patch({'cpu': 1, 'memory': 256, 'replicas':2}, containers, production=True)

    def test_patch_configmap_envvar(self):
        ss = SearchSpaceModel(self.ss)

        for manifest in ss.manifests:
            if isinstance(manifest, ConfigMapSearhSpaceModel) and manifest.name == 'nginx-test-cm-2':
                manifest.patch({'key': 1, 'foo': 'bar-2'}, production=True)

    def test_patch_configmap_jvm(self):
        ss = SearchSpaceModel(self.ss)

        for manifest in ss.manifests:
            if isinstance(manifest, ConfigMapSearhSpaceModel) and manifest.name == 'nginx-test-cm-3':
                manifest.patch({
                    '-Xmx': 256,
                    '-Dhttp.keepalive': True,
                    '-Dhttp.maxConnections': 100,
                    '-Xnojit': True,
                    '-Xnoaot': False,
                    '-XX:+UseContainerSupport': '-XX:+UseContainerSupport',
                    '-Xgcpolicy:gencon': '-Xgcpolicy:gencon'
                }, production=True)

    def test_Option_range(self):
        r = {'name': 'test_name', 'type': 'bool', 'values': ['True', 'False']}
        orm = OptionRangeModel(r)
        self.assertListEqual([True, False], orm.get_values())

        r = {'name': 'test_name', 'type': 'integer', 'values': [1,2,3]}
        orm = OptionRangeModel(r)
        self.assertListEqual([1,2,3], orm.get_values())

        r = {'name': 'test_name', 'type': 'real', 'values': [3.14, 2.71, 1.618]}
        orm = OptionRangeModel(r)
        self.assertListEqual([3.14, 2.71, 1.618], orm.get_values())



if __name__ == '__main__':
    main()