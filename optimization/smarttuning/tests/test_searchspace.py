import os
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import config

from kubernetes.client.models import *
import kubernetes
from controllers import searchspace
from bayesian import BayesianDTO
from controllers.searchspacemodel import SearchSpaceModel
from controllers.k8seventloop import EventLoop, ListToWatch
from controllers.searchspace import SearchSpaceContext
from sampler import Metric

warnings.simplefilter("ignore", ResourceWarning)


class TestSearchSpace(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config.init_k8s(hostname='trxrhel7perf-1')

    def test_get_service(self):
        self.assertTrue(isinstance(searchspace.get_service('acmeair-booking-service', 'default'), V1Service))

    def test_get_deployment(self):
        self.assertTrue(isinstance(searchspace.get_deployment('acmeair-bookingservice', 'default'), V1Deployment))

    def test_instantiate_event_loop(self):
        loop = EventLoop(ThreadPoolExecutor())
        ss = MagicMock()
        ss.namespace = 'default'
        ss.parse_manifests = MagicMock(return_value={'deployment':'', 'name':'', 'service':'', 'namespace': 'default'})
        ctx = SearchSpaceContext(name='test_instantiate_event_loop', search_space=ss)

        list_to_watch = ctx.function_of_observables()
        loop.register(ctx.name, list_to_watch, ctx.selector)
        self.assertListEqual(list(loop.loops.keys()), ['test_instantiate_event_loop'])
        time.sleep(1)
        ctx.delete_bayesian_searchspace()
        loop.unregister(ctx.name)
        loop.shutdown()

    def test_watcher(self):
        client = kubernetes.client.CustomObjectsApi()
        list_to_watch = ListToWatch(client.list_namespaced_custom_object, namespace='default',
                                    group='smarttuning.ibm.com',
                                    version='v1alpha2',
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
        ctx: SearchSpaceContext = searchspace.context('acmeair-booking-ss')

        metric = Metric(cpu=1, memory=1, throughput=1, process_time=1, errors=1, to_eval='cpu')

        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1)  # emulating sampling time
        first_config = ctx.get_from_engine()

        self.assertDictEqual(first_config, ctx.get_best_so_far()[0])
        self.assertEqual(1, ctx.get_best_so_far()[1])

        metric._cpu = 2
        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1)  # emulating sampling time
        new_config = ctx.get_from_engine()

        self.assertNotEqual(first_config, new_config)
        self.assertEqual(1, ctx.get_best_so_far()[1])

        metric._cpu = 0
        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1)  # emulating sampling time
        new_config = ctx.get_from_engine()
        self.assertNotEqual(first_config, new_config)
        self.assertEqual(0, ctx.get_best_so_far()[1])
        self.assertDictEqual(new_config, ctx.get_best_so_far()[0])

        loop.unregister(ctx.name)
        loop.shutdown()

    def test_instantiate_from_event(self):
        loop = EventLoop(ThreadPoolExecutor())
        searchspace.init(loop)

        time.sleep(1)
        self.assertEqual(len(searchspace.search_spaces), 2)
        ctx1: SearchSpaceContext = searchspace.context('acmeair-booking-ss')
        ctx2: SearchSpaceContext = searchspace.context('acmeair-customer-ss')

        metric = Metric(cpu=1, memory=1, throughput=1, process_time=1, errors=1, to_eval='cpu')

        ctx1.put_into_engine(BayesianDTO(metric, ''))
        self.assertGreater(len(ctx1.get_from_engine()), 0)

        ctx2.put_into_engine(BayesianDTO(metric, ''))
        self.assertGreater(len(ctx1.get_from_engine()), 0)

        loop.unregister(ctx1.name)
        loop.unregister(ctx2.name)
        loop.shutdown()

    def test_jvmopt_workflow(self):
        """
        should be executed apart of the other tests
        """
        loop = EventLoop(ThreadPoolExecutor())
        searchspace.init(loop)

        time.sleep(1)
        self.assertGreater(len(searchspace.search_spaces), 0)
        ctx: SearchSpaceContext = searchspace.context('acmeair-customer-ss')

        metric = Metric(cpu=1, memory=1, throughput=1, process_time=1, errors=1, to_eval='cpu')

        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1)  # emulating sampling time
        first_config = ctx.get_from_engine()

        self.assertDictEqual(first_config, ctx.get_best_so_far()[0])
        self.assertEqual(1, ctx.get_best_so_far()[1])

        metric._cpu = 2
        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1)  # emulating sampling time
        new_config = ctx.get_from_engine()

        self.assertNotEqual(first_config, new_config)
        self.assertEqual(1, ctx.get_best_so_far()[1])

        metric._cpu = 0
        ctx.put_into_engine(BayesianDTO(metric, ''))
        time.sleep(1)  # emulating sampling time
        new_config = ctx.get_from_engine()
        self.assertNotEqual(first_config, new_config)
        self.assertEqual(0, ctx.get_best_so_far()[1])
        self.assertDictEqual(new_config, ctx.get_best_so_far()[0])

        loop.shutdown()

    def test_instantiate_from_event(self):
        loop = EventLoop(ThreadPoolExecutor())
        searchspace.init(loop)

        time.sleep(10)
        self.assertTrue(searchspace.context('acmeair-customer-ss'))
        ctx: SearchSpaceContext = searchspace.context('acmeair-customer-ss')

        metric = Metric(cpu=1, memory=1, throughput=1, process_time=1, errors=1, to_eval='cpu')

        import re
        for _ in range(1):
            ctx.put_into_engine(BayesianDTO(metric, ''))
            result = ctx.get_from_engine()
            self.assertGreater(len(result), 0)

            for key, value in result.items():
                for manifest in ctx.model.manifests:
                    if key == manifest._name:
                        self.assertTrue(
                            re.fullmatch(
                                "-Dhttp\.keepalive=(false|true)\s*-Dhttp\.maxConnectionse=[0-9]+\s*-XX:-UseContainerSupport\s*-Xgcpolicy:[a-z]+\s+-Xtune:virtualized",
                                searchspace.dict_to_jvmoptions(manifest.patch(value).data)[0])
                        )

        loop.shutdown()


if __name__ == '__main__':
    unittest.main()
