from __future__ import annotations

import logging

import optuna
from hyperopt import Trials

import config
# workaround to fix circular dependency
# https://www.stefaanlippens.net/circular-imports-type-hints-python.html
from controllers.searchspacemodel import SearchSpaceModel
from models.configuration import Configuration, DefaultConfiguration
from sampler import Metric

logger = logging.getLogger(config.SMARTTUNING_TRIALS_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)


class SmartTuningTrials:
    def __init__(self, space: SearchSpaceModel):
        self.ctx = space.study
        self._space = space
        self._data = {}

    def serialize(self) -> list[dict]:
        documents = []
        i = 0
        try:
            trial: optuna.trial.FrozenTrial
            for i, trial in enumerate(self.wrapped_trials):
                params = trial.params
                loss = trial.value
                status = trial.state.name
                tid = trial.number

                documents.append({
                    'tid': tid,
                    'params': params,
                    'loss': loss,
                    'status': status
                })
        except:
            logger.exception(f'error when retrieving trials at it: {i}')

        return documents

    def last_trial(self) -> optuna.trial.FrozenTrial:
        if len(self.wrapped_trials) > 0:
            return self.wrapped_trials[-1]
        return None
        # uid = self.last_uid()
        # if uid > 0:
        #     return self.wrapped_trials[uid]
        # else:
        #     return None

    def last_uid(self) -> int:
        if len(self.wrapped_trials) > 0:
            return self.wrapped_trials[-1].number
        return 0
        # logger.warning('no trial available')
        # raise IndexError

    @property
    def wrapped_trials(self) -> list[optuna.trial.FrozenTrial]:
        return self.ctx.get_trials(deepcopy=False)

    def add_default_config(self, data: dict, metric: Metric) -> DefaultConfiguration:
        trial = self.new_trial_entry(data, metric.objective(), classification=None)
        default_configuration = DefaultConfiguration(trial=trial, ctx=self._space, trials=self)
        self.add_new_configuration(configuration=default_configuration)
        default_configuration.update_score(metric)
        return default_configuration

    def new_trial_entry(self, configuration: dict, loss: float,
                        classification: str = None) -> optuna.trial.BaseTrial:

        flat_config = {}
        # {'manifest_name': {'x':0}} --> {'x':0}
        to_flatten = list(configuration.items())
        while len(to_flatten) > 0:
            item = to_flatten.pop()
            if isinstance(item, tuple):
                if isinstance(item[1], dict):
                    to_flatten.append(item[1])
            else:
                flat_config.update(item)

        new_trial = optuna.create_trial(
            params=flat_config,
            distributions={name: optuna.distributions.CategoricalDistribution(choices=[param]) for name, param in flat_config.items()},
            value=loss
        )
        self.ctx.add_trial(new_trial)

        return new_trial

    def add_new_configuration(self, configuration: Configuration):
        if not self.get_config_by_name(configuration.name):
            self._data[configuration.name] = configuration
        return self._data[configuration.name]

    def get_config_by_name(self, name: str) -> Configuration:
        return self._data.get(name, None)

    def get_config_by_id(self, uid: int) -> Configuration:
        for name, configuration in self._data.items():
            if uid == configuration.uid:
                return configuration


class EmptySmartTuningTrials(SmartTuningTrials):
    def __init__(self):
        super().__init__({}, Trials())
