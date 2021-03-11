import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import config
from bayesian import BayesianDTO
from controllers import injector
from controllers import searchspace
from controllers.k8seventloop import EventLoop
from controllers.searchspace import SearchSpaceContext


class TestConfigSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.deployment_name = 'daytrader-service'
        config.init_k8s(hostname=config.LOCALHOST)

    def test_instantiation(self):
        loop = EventLoop(ThreadPoolExecutor())
        injector.init(loop)

        time.sleep(1)

        searchspace.init(loop)

        time.sleep(5)
        ctx: SearchSpaceContext = searchspace.context(self.deployment_name)
        self.assertIsNotNone(ctx, None)
        loop.shutdown()

    def test_sample_config(self):
        loop = EventLoop(ThreadPoolExecutor())
        injector.init(loop)

        time.sleep(1)

        searchspace.init(loop)

        time.sleep(5)
        ctx: SearchSpaceContext = searchspace.context(self.deployment_name)
        self.assertIsNotNone(ctx, None)

        new_ctx: SearchSpaceContext = searchspace.search_spaces.get(self.deployment_name, None)
        print(new_ctx.get_from_engine())
        new_ctx.put_into_engine(BayesianDTO())
        print(new_ctx.get_from_engine())
        loop.shutdown()


if __name__ == '__main__':
    unittest.main()
