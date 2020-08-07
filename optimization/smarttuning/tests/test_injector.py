import unittest
from controllers import injector, searchspace

class TestInjector(unittest.TestCase):
    def test_dep_injection(self):
        dep = searchspace.get_deployment('acmeair-bookingservice', 'default')
        injector.inject_proxy_to_deployment({'object': dep})

    def test_dep_duplication(self):
        dep = searchspace.get_deployment('acmeair-bookingservice', 'default')
        injector.duplicate_deployment_for_training(dep)

    def test_workflow(self):
        dep = searchspace.get_deployment('acmeair-bookingservice', 'default')


if __name__ == '__main__':
    unittest.main()
