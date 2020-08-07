import unittest
from concurrent.futures import ThreadPoolExecutor

from controllers.k8seventloop import EventLoop
import controllers.stcontroller as ctl


class StControllerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls)-> None:
        cls.executor = ThreadPoolExecutor()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.executor.shutdown()

    def test_init(self):
        event_loop = EventLoop(self.executor)
        ctl.init(event_loop)
        event_loop.shutdown()

if __name__ == '__main__':
    unittest.main()
