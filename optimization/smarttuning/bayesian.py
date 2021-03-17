from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty

import optuna
from hyperopt import space_eval

import config
from controllers.searchspacemodel import SearchSpaceModel
from models.configuration import Configuration, LastConfig
from models.smartttuningtrials import SmartTuningTrials
from sampler import Metric

random.seed(config.RANDOM_SEED)
logger = logging.getLogger(config.BAYESIAN_LOGGER)
logger.setLevel(logging.DEBUG)


class BayesianChannel:
    channels: dict[str, dict[str, Queue]] = {}
    lock = threading.RLock()

    @staticmethod
    def _erase_queues(channel_name: str, channel: dict[str, Queue]):
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
        if bayesian_id not in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'putting {value} channel.{bayesian_id}.in')
        BayesianChannel.channels[bayesian_id]['in'].put(value)
        logger.debug(f'put {value} channel.{bayesian_id}.in')

    @staticmethod
    def put_out(bayesian_id, value):
        if bayesian_id not in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'putting {value} into channel.{bayesian_id}.out')
        BayesianChannel.channels[bayesian_id]['out'].put(value)
        logger.debug(f'put {value} into channel.{bayesian_id}.out')

    @staticmethod
    def get_in(bayesian_id):
        if bayesian_id not in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'getting value from channel.{bayesian_id}.in')
        value = BayesianChannel.channels[bayesian_id]['in'].get(True)
        logger.debug(f'got value:{value} from channel.{bayesian_id}.in')
        return value

    @staticmethod
    def get_out(bayesian_id):
        if bayesian_id not in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'getting value channel.{bayesian_id}.out')
        value = BayesianChannel.channels[bayesian_id]['out'].get(True)
        logger.debug(f'got value:{value} channel.{bayesian_id}.out')
        return value


class BayesianDTO:
    def __init__(self, metric: Metric = Metric.zero(), workload_classification: str = ''):
        self.metric = metric
        self.classification = workload_classification

    def __repr__(self):
        return f'{{"metric": {self.metric}, "classification": "{self.classification}"}}'


class EmptyBayesianDTO(BayesianDTO):
    def __init__(self):
        super(EmptyBayesianDTO, self).__init__()


@dataclass
class StopCriteria:
    n_iterations_no_change: int = 0
    last_best_score: float = float('inf')

    max_n_no_changes: int = 100

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.BaseTrial):
        self.n_iterations_no_change += 1
        if study.best_value != self.last_best_score:
            self.last_best_score = study.best_value
            self.n_iterations_no_change = 0

        if self.n_iterations_no_change >= self.max_n_no_changes:
            study.stop()


class BayesianEngine:
    # to make search-space dynamic
    # https://github.com/hyperopt/hyperopt/blob/2814a9e047904f11d29a8d01e9f620a97c8a4e37/tutorial/Partial-sampling%20in%20hyperopt.ipynb
    def __init__(self,
                 name: str,
                 space: SearchSpaceModel,
                 max_evals: int,
                 max_evals_no_change: int = 100
                 ):
        self._id: str = name
        self._running: bool = False
        self._space: SearchSpaceModel = space
        self._study: optuna.study.Study = self._space.study
        self._trials: SmartTuningTrials = SmartTuningTrials(space=self._space)
        self._stoped: bool = False
        self._iterations: int = 0
        self._last_best_score: float = float('inf')

        logger.info(f'initializing bayesian engine={name}')

        def objective_fn():
            # TODO: implement eager stop
            # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html?highlight=stop#optuna.study.Study.stop
            try:
                self._study.optimize(
                    self.objective,
                    n_trials=max_evals,
                    show_progress_bar=False,
                    callbacks=[StopCriteria(max_n_no_changes=max_evals_no_change)]
                )

                best_trial = self._study.best_trial
                best = self._study.best_params
                logger.info(f'final best config: {best_trial.params}:{best_trial.value}')
                best.update({'is_best_config': True})

                best_config = self.smarttuning_trials.get_config_by_trial(best_trial)
                # best_config = self.smarttuning_trials.get_config_by_trial(best_trial)
                logger.info(f'found best config after {max_evals} iterations: {best_config}')

                best = LastConfig(best_config)
                BayesianChannel.put_out(self.id(), best)

                # BayesianChannel.put_out(self.id(), LastConfig(trial=best_trial, ctx=self._space, trials=self.smarttuning_trials))
                # BayesianChannel.put_out(self.id(), best)
            except:
                logger.exception('error while evaluating fmin')

        self.fmin = threading.Thread(name='bayesian-engine-' + name, target=objective_fn, daemon=True)
        self.fmin.start()
        self._running = True

    def id(self) -> str:
        return self._id

    def eval_data(self, data: dict):
        logger.info(f'data:{data} -- space:{self._space}')
        return space_eval(self._space, data)

    @property
    def trials(self) -> list[optuna.trial.BaseTrial]:
        return self._trials.wrapped_trials

    @property
    def smarttuning_trials(self) -> SmartTuningTrials:
        return self._trials

    def trials_as_documents(self):
        documents = []

        try:
            trial: optuna.trial.FrozenTrial
            for trial in self.trials.trials:
                params = trial.params
                loss = trial.value
                tid = trial.number
                status = trial.state.name
                iteration = tid

                documents.append({
                    'uid': time.time_ns(),
                    'tid': tid,
                    'params': params,
                    'loss': loss,
                    'iteration': iteration,
                    'status': status
                })
        except:
            logger.exception('error when retrieving trials')

        return documents

    def is_running(self):
        return self._running

    def objective(self, trial: optuna.trial.BaseTrial):
        if self._stoped:
            return float('nan')

        self._iterations += 1

        loss = float('nan')
        try:
            # pass params to configuration through search space model: <configuration.data = space.sample()>
            configuration = Configuration(trial=trial, ctx=self._space, trials=self.smarttuning_trials)
            BayesianChannel.put_out(self.id(), configuration)
            dto: BayesianDTO = BayesianChannel.get_in(self.id())
            # TODO: evaluate intermediary values to prune poor configs
            # TODO: bring 'wait' from 'app.py' here
            configuration.update_score(dto.metric)
            loss = configuration.score
            self._last_best_score = min(self._last_best_score, loss)

        except Exception:
            logger.exception(f'evalution failed at bayesian core for {trial.params}')
            exit(0)
        finally:
            logger.debug(f'params at bayesian obj: {trial.params}:{loss}')
            return loss

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
        return self._study.best_trial.params, self._study.best_value
