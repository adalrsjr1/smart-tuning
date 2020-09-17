import logging
import threading
import time
from functools import partial
from queue import Queue

import numpy as np
import random
from hyperopt import fmin, tpe, rand, Trials, STATUS_OK, STATUS_FAIL, space_eval

import config
from sampler import Metric
random.seed(config.RANDOM_SEED)
logger = logging.getLogger(config.BAYESIAN_LOGGER)
logger.setLevel(logging.DEBUG)


class BayesianChannel:
    channels = {}

    @staticmethod
    def register(bayesian_id):
        BayesianChannel.channels[bayesian_id] = {
            'in': Queue(maxsize=1),
            'out': Queue(maxsize=1)
        }

    @staticmethod
    def unregister(bayesian_id):
        if bayesian_id in BayesianChannel.channels:
            del (BayesianChannel.channels[bayesian_id])

    @staticmethod
    def put_in(bayesian_id, value):
        if not bayesian_id in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'putting {value} channel.{bayesian_id}.in')
        BayesianChannel.channels[bayesian_id]['in'].put(value)

    @staticmethod
    def put_out(bayesian_id, value):
        if not bayesian_id in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'putting {value} into channel.{bayesian_id}.out')
        BayesianChannel.channels[bayesian_id]['out'].put(value)

    @staticmethod
    def get_in(bayesian_id):
        if not bayesian_id in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'getting value from channel.{bayesian_id}.in')
        return BayesianChannel.channels[bayesian_id]['in'].get(True)

    @staticmethod
    def get_out(bayesian_id):
        if not bayesian_id in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'getting value channel.{bayesian_id}.out')
        return BayesianChannel.channels[bayesian_id]['out'].get(True)


class BayesianDTO:
    def __init__(self, metric=Metric.zero(), classification=''):
        self.metric = metric
        self.classification = classification

    def __repr__(self):
        return f'{{"metric": {self.metric}, "classification": "{self.classification}"}}'


class BayesianEngine:
    ## to make search-space dynamic
    ## https://github.com/hyperopt/hyperopt/blob/2814a9e047904f11d29a8d01e9f620a97c8a4e37/tutorial/Partial-sampling%20in%20hyperopt.ipynb
    def __init__(self, name: str, space=None, is_bayesian=True, max_evals=int(1e15)):
        self._id = name
        self._running = False
        self._trials = Trials()
        self._space = space
        self.stoped = False

        surrogate = rand.suggest
        if is_bayesian:
            # n_startup_jobs: # of jobs doing random search at begining of optimization
            # n_EI_candidades: number of config samples draw before select the best. lower number encourages exploration
            # gamma: p(y) in p(y|x) = p(x|y) * p(x)/p(y) or specifically  1/(gamma + g(x)/l(x)(1-gamma))
            surrogate = partial(tpe.suggest, n_startup_jobs=config.N_STARTUP_JOBS,
                                n_EI_candidates=config.N_EI_CANDIDATES, gamma=config.GAMMA)

        # add early_stop when next Hyperopt version came out
        # https://github.com/hyperopt/hyperopt/blob/abf6718951eecc1c43d591f59da321f2de5a8cbf/hyperopt/tests/test_fmin.py#L336
        logger.info(f'initializing bayesian engine={name}')
        np.random.seed(config.RANDOM_SEED)
        self.objective_fn = partial(fmin, fn=self.objective, trials=self.trials(), space=self._space, algo=surrogate,
                                    max_evals=max_evals,
                                    verbose=False, show_progressbar=False,
                                    rstate=np.random.RandomState(random.randint(0,1000)))

        self.fmin = threading.Thread(name='bayesian-engine-' + name, target=self.objective_fn, daemon=True)
        self.fmin.start()
        self._running = True

    def id(self) -> str:
        return self._id

    def trials(self):
        return self._trials

    def is_running(self):
        return self._running

    def objective(self, params):
        logger.debug(f'params at bayesian obj: {params}')
        if self.stoped:
            return {
                'loss': float('inf'),
                'status': STATUS_FAIL,
                # -- store other results like this
                'eval_time': time.time(),
                'classification': None,
                # -- attachments are handled differently
                # https://github.com/hyperopt/hyperopt/wiki/FMin
                # 'attachments':
                #     {'classification': pickle.dumps(classficiation)}
            }

        # follow this hint for implement multiple workload types
        # https://github.com/hyperopt/hyperopt/issues/181
        status = STATUS_FAIL
        loss = float('inf')
        classification = ''
        try:
            BayesianChannel.put_out(self.id(), params)
            dto: BayesianDTO = BayesianChannel.get_in(self.id())
            loss = dto.metric.objective()
            classification = dto.classification
            status = STATUS_OK
        except Exception:
            logger.exception('evalution failed at bayesian core')
        finally:
            return {
                'loss': loss,
                'status': status,
                # -- store other results like this
                'eval_time': time.time(),
                'classification': classification,
                # -- attachments are handled differently
                # https://github.com/hyperopt/hyperopt/wiki/FMin
                # 'attachments':
                #     {'classification': pickle.dumps(classficiation)}
            }

    def stop(self):
        self.stoped = True

    def get(self):
        parameters = BayesianChannel.get_out(self.id())
        return parameters

    def put(self, dto: BayesianDTO):
        BayesianChannel.put_in(self.id(), dto)

    def best_so_far(self) -> (dict, float):
        """
        Be careful!!!! it isn't thread safe

        :return (dict, float) -> (best configuration, best loss)
        """
        loss = self.trials().best_trial['result']['loss']
        best = self.trials().argmin
        return space_eval(self._space, best), loss

    def sample(self, metric):
        parameters = BayesianChannel.get_out(self.id())
        BayesianChannel.put_in(self.id(), metric)
        return parameters, metric
