import unittest

from controllers.planner import Planner
from models.instance import Instance
from unittest.mock import MagicMock

class PlannerTestCase(unittest.TestCase):
    def test_planner(self):
        ctx = MagicMock()

        t = Instance(
            name='daytrader-servicesmarttuning',
            namespace='default',
            is_production=False,
            sample_interval_in_secs=60,
            ctx=ctx
        )
        p = Instance(
            name='daytrader-services',
            namespace='default',
            is_production=False,
            sample_interval_in_secs=60,
            ctx=ctx
        )
        p = Planner(p, t, ctx)

if __name__ == '__main__':
    unittest.main()
