import os
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor

from kubernetes import config as k8sconfig

from bayesian import BayesianDTO
from controllers.k8seventloop import EventLoop
from controllers.searchspace import SearchSpaceContext
from sampler import Metric

warnings.simplefilter("ignore", ResourceWarning)

class TestSearchSpace(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if 'KUBERNETES_SERVICE_HOST' in os.environ:
            k8sconfig.load_incluster_config()
        else:
            k8sconfig.load_kube_config()

    def test_instantiate_event_loop(self):
        loop = EventLoop(ThreadPoolExecutor())
        ctx = SearchSpaceContext(name='test_instantiate_event_loop', namespace='default')

        list_to_watch = ctx.observables()
        loop.register(ctx.name, list_to_watch, ctx.selector)
        time.sleep(1)
        ctx.delete_bayesian_searchspace()
        loop.unregister(ctx.name)

    def test_workflow(self):
        loop = EventLoop(ThreadPoolExecutor())
        ctx = SearchSpaceContext(name='test_workflow', namespace='default')

        metric = Metric(
            cpu=1,
            memory=1,
            throughput=1,
            latency=1,
            errors=1,
            to_eval='cpu'
        )

        list_to_watch = ctx.observables()
        loop.register(ctx.name, list_to_watch, ctx.selector)

        time.sleep(1) # waiting for the first event ADDED

        ctx.put_into_engine(ctx.name, BayesianDTO(metric, ''))
        first_config = ctx.get_from_engine(ctx.name)
        self.assertIsNotNone(first_config)

        ctx.put_into_engine(ctx.name, BayesianDTO(metric, ''))
        self.assertNotEqual(first_config, ctx.get_from_engine(ctx.name))
        self.assertDictEqual(first_config, ctx.get_best_so_far(ctx.name))

        ctx.put_into_engine(ctx.name, BayesianDTO(metric, ''))
        self.assertNotEqual(first_config, ctx.get_from_engine(ctx.name))
        self.assertDictEqual(first_config, ctx.get_best_so_far(ctx.name))
        ctx.delete_bayesian_searchspace()
        loop.unregister(ctx.name)


if __name__ == '__main__':
    unittest.main()
