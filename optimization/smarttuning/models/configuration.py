from __future__ import annotations

import hashlib
import logging
import threading
from typing import TYPE_CHECKING

import optuna

import config
# workaround to fix circular dependency
# https://www.stefaanlippens.net/circular-imports-type-hints-python.html
from controllers.searchspacemodel import SearchSpaceModel

if TYPE_CHECKING:
    from models.smartttuningtrials import SmartTuningTrials
from sampler import Metric
from util.stats import RunningStats

logger = logging.getLogger(config.CONFIGURATION_MODEL_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)


class Configuration:
    def __init__(self, trial: optuna.trial.BaseTrial, ctx: SearchSpaceModel, trials: SmartTuningTrials):
        self._uid = trial.number
        self._trial = trial
        self._trials = trials
        self._ctx = ctx
        self._data = self.ctx.sample(trial, full=True)
        self._name = hashlib.md5(bytes(str(self.data.items()), 'ascii')).hexdigest()
        self.stats: RunningStats = RunningStats()
        self._n_restarts = 0
        self._workload = ''
        self._was_pruned = False
        self.semaphore = threading.BoundedSemaphore(value=1)
        self._lock = threading.Lock()

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other: Configuration):
        return self.stats.median() < other.stats.median()

    def __gt__(self, other: Configuration):
        return self.stats.median() > other.stats.median()

    def __eq__(self, other: Configuration):
        # return self.stats == other.stats
        return self.name == other.name

    def __str__(self):
        return f'{{' \
               f'"name":{self.name}, ' \
               f'"uid":{self.uid}, ' \
               f'"pruned":{self.was_pruned}, ' \
               f'"workload": {self.workload}, ' \
               f'"score":{self.score}, ' \
               f'"mean":{self.mean()}, ' \
               f'"std":{self.stddev()}, ' \
               f'"median":{self.median()}' \
               f'}}'

    def __repr__(self):
        return self.__str__()

    def serialize(self) -> dict:
        logger.debug(f'serializing: {id(self)}:{self}')
        return {
            'uid': self.uid,
            'name': self.name,
            'pruned': self.was_pruned,
            'workload': self.workload,
            'data': self.data,
            'score': self.score,
            'stats': self.stats.serialize(),
            'trials': self._trials.serialize(),
            'restarts': self.n_restarts,
        }

    def prune(self):
        with self._lock:
            logger.warning(f'pruning config: {self.name} -- {id(self)}')
            self._was_pruned = True
            logger.debug(f'pruning config: {self.name} -- {self._was_pruned}')

    @property
    def was_pruned(self):
        with self._lock:
            return self._was_pruned

    def restore(self):
        with self._lock:
            self._was_pruned = False

    @property
    def n_restarts(self):
        return self._n_restarts

    def increment_restart_counter(self):
        self._n_restarts += 1

    @property
    def uid(self):
        return self.trial.number

    @property
    def ctx(self):
        return self._ctx

    @property
    def trial(self):
        return self._trial

    @trial.setter
    def trial(self, trial):
        self._trial = trial

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def score(self):
        return self.stats.curr()

    def update_score(self, value: Metric):
        self.stats.push(value.objective())
        self.trial.value = self.median()

    @property
    def workload(self):
        return self._workload

    @workload.setter
    def workload(self, new_workload):
        self._workload = new_workload

    def mean(self):
        return self.stats.mean()

    def stddev(self):
        return self.stats.standard_deviation()

    def median(self):
        return self.stats.median()

    def iterations(self):
        self.stats.n()


class DefaultConfiguration(Configuration):
    def __init__(self, trial: optuna.trial.BaseTrial, ctx: SearchSpaceModel, trials: SmartTuningTrials,
                 workload: str):
        self._uid = trial.number
        self._trial = trial
        self._trials = trials
        self._ctx = ctx
        self._data = trial.params
        self._name = hashlib.md5(bytes(str(self.data.items()), 'ascii')).hexdigest()
        self.stats = RunningStats()
        self._n_restarts = 0
        self._workload = workload
        self._was_pruned = False
        self._lock = threading.Lock()
        self.semaphore = threading.BoundedSemaphore(value=1)

    @property
    def data(self):
        return self.ctx.default_structure(self._data)


class EmptyConfiguration(Configuration):
    def __init__(self):
        self._uid = -1
        self._trial = None
        self._trials = None
        self._ctx = None
        self._data = {}
        self._name = hashlib.md5(bytes(str(self.data.items()), 'ascii')).hexdigest()
        self.stats = RunningStats()
        self._n_restarts = 0
        self.workload = ''
        self._was_pruned = False
        self._lock = threading.Lock()
        self.semaphore = threading.BoundedSemaphore(value=1)

    def __str__(self):
        return 'EmptyConfig'

    def serialize(self) -> dict:
        return {}


class LastConfig(Configuration):
    def __init__(self, last_config: Configuration):
        self._uid = last_config.uid
        self._trial = last_config.trial
        self._trials = last_config._trials
        self._ctx = last_config.ctx
        self._data = last_config.data
        self._name = last_config.name
        self.stats = last_config.stats
        self._n_restarts = last_config.n_restarts
        self.workload = last_config.workload
        self._was_pruned = False
        self._lock = threading.Lock()
        self.semaphore = threading.BoundedSemaphore(value=1)
