import unittest
from kubernetes.client.models import *
import kubernetes
from controllers import searchspace
from bayesian import BayesianDTO
from controllers.searchspacemodel import SearchSpaceModel
from controllers.k8seventloop import EventLoop, ListToWatch
from controllers.searchspace import SearchSpaceContext, get_deployment
from controllers import  injector
import config
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import time

class TestConfigSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config.init_k8s(hostname='trxrhel7perf-1')

    def test_instantiation(self):
        loop = EventLoop(ThreadPoolExecutor())
        injector.init(loop)

        time.sleep(1)

        searchspace.init(loop)

        time.sleep(5)
        ctx: SearchSpaceContext = searchspace.context('acmeair-bookingservice')
        self.assertIsNotNone(ctx, None)
        print(ctx.model.search_space())
        loop.shutdown()

    def test_sample_config(self):
        loop = EventLoop(ThreadPoolExecutor())
        injector.init(loop)

        time.sleep(1)

        searchspace.init(loop)

        time.sleep(5)
        ctx: SearchSpaceContext = searchspace.context('acmeair-bookingservice')
        self.assertIsNotNone(ctx, None)

        new_ctx: SearchSpaceContext = searchspace.search_spaces.get('acmeair-bookingservice',None)
        print(new_ctx.get_from_engine())
        new_ctx.put_into_engine(BayesianDTO())
        print(new_ctx.get_from_engine())
        loop.shutdown()

if __name__ == '__main__':
    unittest.main()
