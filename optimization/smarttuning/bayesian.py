from __future__ import annotations

import copy
import logging
import random
import threading
from dataclasses import dataclass
from queue import Queue, Empty

import optuna
from hyperopt import space_eval
from optuna.pruners import BasePruner

import config
from controllers.searchspacemodel import SearchSpaceModel
from models.bayesiandto import BayesianDTO
from models.configuration import Configuration, LastConfig, EmptyConfiguration
from models.smartttuningtrials import SmartTuningTrials

random.seed(config.RANDOM_SEED)
logger = logging.getLogger(config.BAYESIAN_LOGGER)
logger.setLevel(logging.DEBUG)


class SnapshotQueue(Queue):
    def snapshot(self):
        with self.mutex:
            return list(self.queue)


class BayesianChannel:
    channels: dict[str, dict[str, SnapshotQueue]] = {}
    lock = threading.RLock()

    @staticmethod
    def _erase_queues(channel_name: str, channel: dict[str, SnapshotQueue]):
        gate: SnapshotQueue
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
            'in': SnapshotQueue(maxsize=1),
            'out': SnapshotQueue(maxsize=1)
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

    @staticmethod
    def head(bayesian_id):
        if bayesian_id not in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'sneaking value channel.{bayesian_id}.out')
        nqueue = BayesianChannel.channels[bayesian_id]['out'].snapshot()
        logger.debug(f'sneaked value:{nqueue[0]} channel.{bayesian_id}.out')
        return copy.deepcopy(nqueue[0])

    @staticmethod
    def task_done(bayesian_id):
        if bayesian_id not in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'task done value channel.{bayesian_id}.out')
        BayesianChannel.channels[bayesian_id]['out'].task_done()
        logger.debug(f'done task  channel.{bayesian_id}.out')

    @staticmethod
    def join(bayesian_id):
        if bayesian_id not in BayesianChannel.channels:
            BayesianChannel.register(bayesian_id)
        logger.debug(f'join value channel.{bayesian_id}.out')
        BayesianChannel.channels[bayesian_id]['out'].join()
        logger.debug(f'joined  channel.{bayesian_id}.out')


class SmartTuningPrunner(BasePruner):

    def __init__(self, workload: str, ctx):
        super(SmartTuningPrunner, self).__init__()
        self._workload: str = workload
        self._ctx = ctx
        self._smart_tuning_trials: SmartTuningTrials = self._ctx.get_smarttuning_trials()

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        from controllers.planner import Planner
        configuration: Configuration = self._smart_tuning_trials.get_config_by_trial(trial)
        pruned = configuration.workload != Planner.get_workload(self._ctx.deployment)
        if pruned:
            configuration.prune()
            study.enqueue_trial(trial.params)
        return pruned


@dataclass
class StopCriteria:
    n_iterations_no_change: int = 0
    last_best_score: float = float('inf')
    last_best_uid: int = -1
    last_best_config: Configuration = EmptyConfiguration()
    smart_tuning_trials: SmartTuningTrials = None
    max_n_no_changes: int = 100

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.BaseTrial):
        try:
            logger.info(f'stop criteria @ {study.study_name} -- n_iterations_no_change:{self.n_iterations_no_change}/{self.max_n_no_changes}, '
                        f'last_best_score|id: {self.last_best_score}|{self.last_best_uid}, '
                        f'best_config|workload: {self.last_best_config.name}|{self.last_best_config.workload}')
        except Exception:
            logger.info(f'stop criteria @ {study.study_name} -- n_iterations_no_change:{self.n_iterations_no_change}/{self.max_n_no_changes}, '
                        f'last_best_score|id: {self.last_best_score}|{self.last_best_uid}')

        self.n_iterations_no_change += 1

        if study.best_value != self.last_best_score:

            self.last_best_score = study.best_trial.value
            self.last_best_uid = study.best_trial.number
            self.last_best_config = self.smart_tuning_trials.get_config_by_trial(study.best_trial)
            self.n_iterations_no_change = 0

        if self.n_iterations_no_change >= self.max_n_no_changes:
            logger.warning(f'stoping iterations at study: {study.study_name}')
            logger.warning(
                f'n_iterations_no_change:{self.n_iterations_no_change} max_n_no_chanages:{self.max_n_no_changes}')
            study.stop()


class BayesianEngine:
    def __init__(self,
                 name: str,
                 space: SearchSpaceModel,
                 workload: str,
                 max_evals: int,
                 max_evals_no_change: int = 100
                 ):
        self._id: str = name
        self._running: bool = False
        self._workload: str = workload
        self._space: SearchSpaceModel = space
        self._study: optuna.study.Study = self._space.study
        self._trials: SmartTuningTrials = SmartTuningTrials(space=self._space)
        self._stoped: bool = False
        self._iterations: int = 0
        self._last_best_score: float = float('inf')

        logger.info(f'initializing bayesian engine={name}')

        def objective_fn():
            try:
                self._study.optimize(
                    self.objective,
                    n_trials=max_evals,
                    show_progress_bar=False,
                    callbacks=[StopCriteria(max_n_no_changes=max_evals_no_change, smart_tuning_trials=self.smarttuning_trials)]
                )

                best_trial = self._study.best_trial
                best = self._study.best_params
                logger.info(f'final best config: {best_trial.params}:{best_trial.value}')
                best.update({'is_best_config': True})

                best_config = self.smarttuning_trials.get_config_by_trial(best_trial)
                logger.info(f'found best config after {max_evals} iterations: {best_config}')

                best = LastConfig(best_config)
                BayesianChannel.put_out(self.id(), best)
                self.stop()
            except Exception:
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

    @property
    def workload(self):
        return self._workload

    # TODO: to remove
    # def trials_as_documents(self):
    #     documents = []
    #
    #     try:
    #         trial: optuna.trial.FrozenTrial
    #         for trial in self.trials.trials:
    #             params = trial.params
    #             loss = trial.value
    #             tid = trial.number
    #             status = trial.state.name
    #             iteration = tid
    #
    #             documents.append({
    #                 'uid': time.time_ns(),
    #                 'tid': tid,
    #                 'params': params,
    #                 'loss': loss,
    #                 'iteration': iteration,
    #                 'status': status
    #             })
    #     except Exception:
    #         logger.exception('error when retrieving trials')
    #
    #     return documents

    def is_running(self):
        return self._running

    def objective(self, trial: optuna.trial.BaseTrial):
        if self._stoped:
            return float('nan')

        self._iterations += 1

        loss = float('nan')
        configuration = None
        try:
            configuration = self.smarttuning_trials.get_config_by_trial(trial)
            if configuration is None:
                configuration = Configuration(trial=trial, ctx=self._space, trials=self.smarttuning_trials)
                configuration.workload = self.workload
                self.smarttuning_trials.add_new_configuration(configuration)

            # out: config
            BayesianChannel.put_out(self.id(), configuration)
            # in: score
            dto: BayesianDTO = BayesianChannel.get_in(self.id())

            if not configuration.was_pruned:
                # don't update config score if it was pruned
                configuration.update_score(dto.metric)
                # configuration.workload = dto.classification
                loss = configuration.score
                self._last_best_score = min(self._last_best_score, loss)

        except Exception:
            logger.exception(f'evalution failed at bayesian core for {self._study.study_name}:{trial.params}')
            logger.warning(f'going to prune due to error: {configuration}')
        finally:
            with configuration.semaphore:
                if not (configuration.trial is trial):
                    # updating trial if reusing a config
                    configuration.trial = trial

            if configuration.was_pruned:
                logger.warning(f'pruning')
                raise optuna.TrialPruned

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

    def task_done(self):
        BayesianChannel.task_done(self.id())

    def join(self):
        BayesianChannel.join(self.id())

    def best_so_far(self) -> (dict, float):
        return self._study.best_trial.params, self._study.best_value
