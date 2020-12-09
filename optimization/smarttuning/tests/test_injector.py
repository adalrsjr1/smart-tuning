import unittest
import config
from controllers import injector, searchspace
from controllers.k8seventloop import EventLoop

class TestInjector(unittest.TestCase):
    def test_dep_injection(self):
        dep = searchspace.get_deployment('daytrader-service', 'default')
        injector.inject_proxy_to_deployment({'object': dep})

    def test_dep_duplication(self):
        dep = searchspace.get_deployment('acmeair-bookingservice', 'default')
        injector.duplicate_deployment_for_training(dep)

    def test_workflow(self):
        dep = searchspace.get_deployment('acmeair-bookingservice', 'default')

    def test_init_injector(self):
        loop = EventLoop(config.executor())
        injector.init(loop)


if __name__ == '__main__':
    unittest.main()
