import copy
import os
import random
import sys
import time
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from hyperopt import Trials, STATUS_OK

from controllers.planner import Configuration
from controllers.planner import Planner
from models.instance import Instance
from models.smartttuningtrials import SmartTuningTrials
from sampler import Metric


def metric(production=True):
    # pmetrics = sampler_pmetric()
    # tmetrics = sampler_tmetric()
    def anonymous():
        # raw_metric = next(pmetrics) if production else next(tmetrics)
        return Metric(
            name='',
            # cpu=raw_metric['cpu'],
            # memory=raw_metric['memory'],
            # throughput=raw_metric['throughput'],
            # process_time=raw_metric['process_time'],
            # errors=raw_metric['errors'],
            to_eval=f'{random.uniform(-random.uniform(500,300), 0)}'
        )

    return anonymous


def smarttuning_trials():
    trials = Trials()
    strials = SmartTuningTrials(space={}, trials=trials)
    strials.new_hyperopt_trial_entry(configuration={}, loss=0, status=STATUS_OK, classification='')
    return strials


def get_current_config():
    # cfg = sampler_pconfig()
    def anonymous():
        # next_cfg = next(cfg)
        # return next_cfg, next_cfg
        return {'name': time.time()}, {'name': time.time()}

    return anonymous


def get_from_engine():
    # cfg = sampler_tconfig()
    def anonymous():
        # next_config = next(cfg)
        # return Configuration(data=next_config, trials=smarttuning_trials())
        return Configuration(data={'name': time.time()}, trials=smarttuning_trials())

    return anonymous


counter = 0


def wait_for_metrics(planner: Planner):
    # pmetrics = next(sampler_pmetric())
    # tmetrics = sampler_tmetric()

    memo = {}

    def anonymous(*kwargs):
        # traw_metric = next(tmetrics)
        # import random
        # t = Metric(
        #     name=planner.training.configuration.name,
        #     # cpu=traw_metric['cpu'],
        #     # memory=traw_metric['memory'],
        #     # throughput=traw_metric['throughput'],
        #     # process_time=traw_metric['process_time'],
        #     # errors=traw_metric['errors'],
        #     to_eval=f'{random.uniform(-300, 0)}'
        # )
        # if t.name not in memo:
        #     memo[id(planner.training.configuration)] = t
        #
        # try:
        #     p = memo[id(planner.training.configuration)]
        # except:
        #     p = Metric.zero()
        # return t, p
        return Metric(to_eval=f'{random.uniform(-300, 0)}'), Metric(to_eval=f'{random.uniform(-300, 0)}')

    return anonymous


class TestPlanner(TestCase):
    def test_planner(self):
        ctx = MagicMock()
        ctx.get_current_config = get_current_config()
        ctx.get_from_engine = get_from_engine()

        tsampler = MagicMock()
        # tsampler.metric = metric(production=False)

        psampler = MagicMock()
        # psampler.metric = metric(production=True)

        t = Instance(
            name='daytrader-servicesmarttuning',
            namespace='default',
            is_production=False,
            sample_interval_in_secs=0,
            ctx=ctx,
            sampler=tsampler
        )
        t.restart = MagicMock(return_value=None)
        p = Instance(
            name='daytrader-services',
            namespace='default',
            is_production=False,
            sample_interval_in_secs=0,
            ctx=ctx,
            sampler=psampler
        )
        p.restart = MagicMock(return_value=None)

        planner = Planner(p, t, ctx, k=10, ratio=0.3334)
        planner.save_trace = MagicMock()
        planner.wait_for_metrics = wait_for_metrics(planner)

        # tmetrics = sampler_tmetric()
        # cfg = sampler_tconfig()
        # while True:
        #     print(next(cfg)['name'], next(tmetrics)['objective'])

        results = []
        try:
            for _ in range(1000):
                c: tuple[Configuration, bool] = copy.deepcopy(next(planner))
                results.append(f'{c[0].name}, {c[0].score:.2f}, {c[0].median():.2f}')
        finally:
            from pprint import pprint
            pprint(results)


if __name__ == '__main__':
    def reset_seeds():
        random.seed(123)


    reset_seeds()

    SEED = 0
    hashseed = os.getenv('PYTHONHASHSEED')
    if not hashseed:
        os.environ['PYTHONHASHSEED'] = str(SEED)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    unittest.main()
