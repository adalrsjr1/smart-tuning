import unittest
import config
from controllers import injector, searchspace
from controllers.k8seventloop import EventLoop

class TestInjector(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.deployement_name = 'daytrader-service'
        cls.namespace = 'default'


    def test_dep_injection(self):
        dep = searchspace.get_deployment(self.deployement_name, self.namespace)
        injector.inject_proxy_to_deployment({'object': dep})

    def test_dep_duplication(self):
        dep = searchspace.get_deployment(self.deployement_name, self.namespace)
        injector.duplicate_deployment_for_training(dep)

    def test_workflow(self):
        dep = searchspace.get_deployment(self.deployement_name, self.namespace)

    def test_init_injector(self):
        loop = EventLoop(config.executor())
        injector.init(loop)


if __name__ == '__main__':
    unittest.main()
