from __future__ import annotations

import copy
import hashlib
import logging
from typing import TYPE_CHECKING

import config

# workaround to fix circular dependency
# https://www.stefaanlippens.net/circular-imports-type-hints-python.html
if TYPE_CHECKING:
    from models.smartttuningtrials import SmartTuningTrials
from sampler import Metric
from util.stats import RunningStats

logger = logging.getLogger(config.CONFIGURATION_MODEL_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)

class Configuration:
    def __init__(self, data:dict, trials: SmartTuningTrials):
        self._name = hashlib.md5(bytes(str(data.items()), 'ascii')).hexdigest()
        self._data = data
        self._current_score = float('inf')
        self.stats = RunningStats()
        self._trials = trials
        self._uid = self._trials.last_uid()

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other:Configuration):
        return self.stats.median() < other.stats.median()

    def __gt__(self, other:Configuration):
        return self.stats.median() > other.stats.median()

    def __eq__(self, other:Configuration):
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
            'trials': self._trials.serialize()
        }

    @property
    def uid(self):
        return self._uid

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def score(self):
        return self._current_score

    def update_score(self, value:Metric, ):
        self._current_score = value.objective()
        self.stats.push(self._current_score)
        self._trials.update_hyperopt_score(self)

    def mean(self):
        return self.stats.mean()

    def stddev(self):
        return self.stats.standard_deviation()

    def median(self):
        return self.stats.median()

    def iterations(self):
        self.stats.n()

class DefaultConfiguration(Configuration):
    def __init__(self, data:dict, valued_data:dict, trials: SmartTuningTrials):
        super().__init__(data, trials)
        # self._name = hashlib.md5(bytes(str(data.items()), 'ascii')).hexdigest()
        # self._data = data
        self._vdata = valued_data
        # self._current_score = float('inf')
        # self.stats = RunningStats()
        # self._trials = trials
        self._uid = len(self._trials.wrapped_trials.trials)

    @property
    def vdata(self):
        return self._vdata

    def serialize(self) -> dict:
        return {
            'uid': self.uid,
            'name': self.name,
            'data': self.vdata,
            'score': self.score,
            'stats': self.stats.serialize(),
            'trials': self._trials.serialize()
        }

class EmptyConfiguration(Configuration):
    def __init__(self):
        from models.smartttuningtrials import EmptySmartTuningTrials
        super().__init__({}, EmptySmartTuningTrials())

class LastConfig(Configuration):
    def __init__(self, data:dict, trials: SmartTuningTrials):
        super().__init__(data, trials)