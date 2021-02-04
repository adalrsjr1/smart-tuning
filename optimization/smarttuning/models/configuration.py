from __future__ import annotations

import hashlib
import logging
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
    def __init__(self, trial: optuna.trial.FrozenTrial, ctx:SearchSpaceModel, trials: SmartTuningTrials):
        self._uid = trial.number
        self._trial = trial
        self._trials = trials
        self._ctx = ctx
        self._data = self.ctx.sample(trial, full=True)
        self._name = hashlib.md5(bytes(str(self.data.items()), 'ascii')).hexdigest()
        self.stats = RunningStats()
        self._n_restarts = 0

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
        return f'{{"name":{self.name}, "uid":{self.uid}, "score":{self.score}, "mean":{self.mean()}, "std":{self.stddev()}, "median":{self.median()}}}'

    def __repr__(self):
        return self.__str__()

    def serialize(self) -> dict:
        return {
            'uid': self.uid,
            'name': self.name,
            'data': self.data,
            'score': self.score,
            'stats': self.stats.serialize(),
            'trials': self._trials.serialize(),
            'restarts': self.n_restarts,
        }

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

    def mean(self):
        return self.stats.mean()

    def stddev(self):
        return self.stats.standard_deviation()

    def median(self):
        return self.stats.median()

    def iterations(self):
        self.stats.n()


class DefaultConfiguration(Configuration):
    def __init__(self, trial: optuna.trial.FrozenTrial, ctx:SearchSpaceModel, trials: SmartTuningTrials):
        self._uid = trial.number
        self._trial = trial
        self._trials = trials
        self._ctx = ctx
        self._data = trial.params
        self._name = hashlib.md5(bytes(str(self.data.items()), 'ascii')).hexdigest()
        self.stats = RunningStats()
        self._n_restarts = 0

    @property
    def data(self):
        return self.ctx.default_structure(self._data)


class EmptyConfiguration(Configuration):
    def __init__(self):
        pass


class LastConfig(Configuration):
    def __init__(self, trial: optuna.trial.Trial, ctx: SearchSpaceModel, trials: SmartTuningTrials):
        super(LastConfig, self).__init__(trial, ctx, trials)

