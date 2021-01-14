from __future__ import annotations

import logging
import random
import threading
import time
from collections import Counter
from functools import partial
from queue import Queue, Empty

import numpy as np
from hyperopt import fmin, tpe, rand, Trials, STATUS_OK, STATUS_FAIL, space_eval
from hyperopt.early_stop import no_progress_loss
from hyperopt.exceptions import AllTrialsFailed

import config
from models.configuration import Configuration
from sampler import Metric

random.seed(config.RANDOM_SEED)
logger = logging.getLogger(config.BAYESIAN_LOGGER)
logger.setLevel(logging.DEBUG)


class BayesianChannel:
    channels:dict[str,dict[str, Queue]] = {}
    lock = threading.RLock()

    @staticmethod
    def _erase_queues(channel_name:str, channel:dict[str, Queue]):
        gate: Queue
        for name, gate in channel.items():
            # with gate.mutex:
            if not gate.empty():
                try:
                    gate.get_nowait()
                except Empty:
                    logger.warning(f'trying to erase the empty queue {channel_name}:{name}')

    @staticmethod
    def register(bayesian_id):
        BayesianChannel.lock.acquire()
        BayesianChannel.channels[bayesian_id] = {
            'in': Queue(maxsize=1),
            'out': Queue(maxsize=1)
        }
        BayesianChannel.lock.release()

    @staticmethod
    def unregister(bayesian_id):
        BayesianChannel.lock.acquire()
        if bayesian_id in BayesianChannel.channels:
            channel = BayesianChannel.channels[bayesian_id]

            BayesianChannel._erase_queues(bayesian_id, channel)

            BayesianChannel._erase_queues(bayesian_id, channel)
            del (BayesianChannel.channels[bayesian_id])
        BayesianChannel.lock.release()

    @staticmethod
    def put_in(bayesian_id, value):
        if not bayesian_id in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'putting {value} channel.{bayesian_id}.in')
        BayesianChannel.channels[bayesian_id]['in'].put(value)
        logger.debug(f'put {value} channel.{bayesian_id}.in')


    @staticmethod
    def put_out(bayesian_id, value):
        if not bayesian_id in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'putting {value} into channel.{bayesian_id}.out')
        BayesianChannel.channels[bayesian_id]['out'].put(value)
        logger.debug(f'put {value} into channel.{bayesian_id}.out')


    @staticmethod
    def get_in(bayesian_id):
        if not bayesian_id in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'getting value from channel.{bayesian_id}.in')
        value = BayesianChannel.channels[bayesian_id]['in'].get(True)
        logger.debug(f'got value:{value} from channel.{bayesian_id}.in')
        return  value

    @staticmethod
    def get_out(bayesian_id):
        if not bayesian_id in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'getting value channel.{bayesian_id}.out')
        value = BayesianChannel.channels[bayesian_id]['out'].get(True)
        logger.debug(f'got value:{value} channel.{bayesian_id}.out')
        return value


class BayesianDTO:
    def __init__(self, metric:Metric=Metric.zero(), workload_classification:str= ''):
        self.metric = metric
        self.classification = workload_classification

    def __repr__(self):
        return f'{{"metric": {self.metric}, "classification": "{self.classification}"}}'

class EmptyBayesianDTO(BayesianDTO):
    def __init__(self):
        super().__init__()


from models.smartttuningtrials import SmartTuningTrials


class BayesianEngine:
    ## to make search-space dynamic
    ## https://github.com/hyperopt/hyperopt/blob/2814a9e047904f11d29a8d01e9f620a97c8a4e37/tutorial/Partial-sampling%20in%20hyperopt.ipynb
    def __init__(self, name: str, space=None, is_bayesian=True, max_evals=config.NUMBER_ITERATIONS):
        self._id = name
        self._running = False
        self._space = space
        self._trials = SmartTuningTrials(space=self._space, trials=Trials())
        self._stoped = False
        self.counter = Counter()

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

        def stop_iterations(trial, count=0):
            return self._stoped, [count + 1]

        def objective_fn():
            try:
                partial_fmin = partial(fmin, fn=self.objective, trials=self.trials, space=self._space, algo=surrogate,
                        max_evals=max_evals,
                        verbose=False, show_progressbar=False,
                        rstate=np.random.RandomState(random.randint(0, 1000)),
                        early_stop_fn=stop_iterations)

                best = partial_fmin()
                best = self.eval_data(best)
                logger.info(f'final best config: {best}')
                best.update({'is_best_config':True})

                best_config = Configuration(data=best, trials=self.smarttuning_trials)
                logger.info(f'found best config after {max_evals} iterations: {best_config}')
                BayesianChannel.put_out(self.id(), best_config)
                # BayesianChannel.put_out(self.id(), best)
            except:
                logger.exception('error while evaluating fmin')
                from pprint import pprint
                pprint(self.trials.trials)
                pprint(self._space)

        self.fmin = threading.Thread(name='bayesian-engine-' + name, target=objective_fn, daemon=True)
        self.fmin.start()
        self._running = True

    def id(self) -> str:
        return self._id

    def eval_data(self, data:dict):
        logger.info(f'data:{data} -- space:{self._space}')
        return space_eval(self._space, data)

    @property
    def trials(self) -> Trials:
        return self._trials.wrapped_trials

    @property
    def smarttuning_trials(self) -> SmartTuningTrials:
        return self._trials

    def trials_as_documents(self):
        documents = []
        try:
            for trial in self.trials().trials:
                params = self.eval_data({k: v[0] for k, v in trial['misc'].get('vals', {}).items()})
                tid = trial['misc'].get('tid', -1)
                loss = trial['result'].get('loss', float('inf'))
                status = trial['result'].get('status', None)
                iteration = trial['result'].get('iteration', -1)

                documents.append({
                    'uid': time.time_ns(),
                    'tid': tid,
                    'params': params,
                    'loss': loss,
                    'iteration':iteration,
                    'status': status
                })
        except:
            logger.exception('error when retrieving trials')

        return documents


    def is_running(self):
        return self._running

    def objective(self, params):
        logger.debug(f'params at bayesian obj: {params}')
        if self._stoped:
            return {
                'iteration': 1,
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
            configuration = Configuration(data=params, trials=self.smarttuning_trials)
            BayesianChannel.put_out(self.id(), configuration)
            dto: BayesianDTO = BayesianChannel.get_in(self.id())
            loss = dto.metric.objective()
            if not isinstance(dto, EmptyBayesianDTO):
                classification = dto.classification
                status = STATUS_OK
        except Exception:
            logger.exception('evalution failed at bayesian core')
        finally:
            return {
                'iteration': 1,
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
        logger.warning('stopping bayesian engine')
        self._stoped = True
        self._running = False

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
        return self.eval_data(best), loss

    def update_best_trial(self, value:float) -> float:
        try:
            # best_idx = list(self.trials().best_trial['misc']['idxs'].values())[0][0]
            # get best trial id
            try:
                best_idx = self.trials().best_trial['tid']
            except AllTrialsFailed:
                logger.debug('empty trials')
                return float('nan')

            # update the current config with the average of the all values
            curr_trial = self.trials().trials[best_idx]
            # TODO: remove all entries related to self.counter
            # self.counter[best_idx] += 1

            curr_trial['result']['iteration'] += 1
            curr = curr_trial['result']['loss']
            # curr = curr + (value - curr) / self.counter[best_idx]
            curr = curr + (value - curr) / curr_trial['result']['iteration']
            curr_trial['result']['loss'] = curr


            #
            # for i, trial in enumerate(self.trials()):
            #     if i == best_idx:
            #         curr = trial['result']['loss']
            #
            #         curr = curr + (value - curr)/self.counter['tid']
            #
            #         trial['result']['loss'] = curr
            #
            #         logger.info(f'updating best config loss to {value}')
            #         break

            # udpates database
            self.trials().refresh()
            return curr
        except:
            logger.exception('erros to access the best trial')
            exit(1)

    def sample(self, metric):
        parameters = BayesianChannel.get_out(self.id())
        BayesianChannel.put_in(self.id(), metric)
        return parameters, metric
