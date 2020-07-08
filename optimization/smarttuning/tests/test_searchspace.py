import os
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor

from kubernetes import config as k8sconfig
import kubernetes
from controllers import searchspace
from bayesian import BayesianDTO
from controllers.k8seventloop import EventLoop, ListToWatch
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

        list_to_watch = ctx.function_of_observables()
        loop.register(ctx.name, list_to_watch, ctx.selector)
        self.assertListEqual(list(loop.loops.keys()), ['test_instantiate_event_loop'])
        time.sleep(1)
        ctx.delete_bayesian_searchspace()
        loop.unregister(ctx.name)

    def test_watcher(self):
        client = kubernetes.client.CustomObjectsApi()
        list_to_watch = ListToWatch(client.list_namespaced_custom_object, namespace='default',
                    group='smarttuning.ibm.com',
                    version='v1alpha1',
                    plural='searchspaces')

        w = kubernetes.watch.Watch()
        t = list_to_watch.fn()
        fn = t[0]
        args = t[1]

        for event in w.stream(fn, **args):
            self.assertEqual(event['type'], 'ADDED')
            w.stop()

    def test_workflow(self):
        """
        should be executed apart of the other tests
        """
        loop = EventLoop(ThreadPoolExecutor())
        searchspace.init(loop)

        time.sleep(1)
        self.assertGreater(len(searchspace.search_spaces), 0)
        ctx: SearchSpaceContext = searchspace.context('test')

        metric = Metric(
            cpu=1,
            memory=1,
            throughput=1,
            process_time=1,
            errors=1,
            to_eval='cpu'
        )

        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1) # emulating sampling time
        first_config = ctx.get_from_engine()

        self.assertDictEqual(first_config, ctx.get_best_so_far()[0])
        self.assertEqual(1, ctx.get_best_so_far()[1])

        metric._cpu = 2
        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1) # emulating sampling time
        new_config = ctx.get_from_engine()

        self.assertNotEqual(first_config, new_config)
        self.assertEqual(1, ctx.get_best_so_far()[1])

        metric._cpu = 0
        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1) # emulating sampling time
        new_config = ctx.get_from_engine()
        self.assertNotEqual(first_config, new_config)
        self.assertEqual(0, ctx.get_best_so_far()[1])
        self.assertDictEqual(new_config, ctx.get_best_so_far()[0])



    def test_instantiate_from_event(self):
        loop = EventLoop(ThreadPoolExecutor())
        searchspace.init(loop)

        time.sleep(1)
        self.assertEqual(len(searchspace.search_spaces), 2)
        ctx1:SearchSpaceContext = searchspace.context('test')
        ctx2:SearchSpaceContext = searchspace.context('test2')

        metric = Metric(
            cpu=1,
            memory=1,
            throughput=1,
            process_time=1,
            errors=1,
            to_eval='cpu'
        )

        ctx1.put_into_engine(BayesianDTO(metric, ''))
        self.assertGreater(len(ctx1.get_from_engine()),0)

        ctx2.put_into_engine(BayesianDTO(metric, ''))
        self.assertGreater(len(ctx1.get_from_engine()), 0)


if __name__ == '__main__':
    unittest.main()
