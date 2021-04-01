from __future__ import annotations

import hashlib
import logging

import optuna

import config
# workaround to fix circular dependency
# https://www.stefaanlippens.net/circular-imports-type-hints-python.html
from controllers.searchspacemodel import SearchSpaceModel
from util.stats import RunningStats

logger = logging.getLogger(config.CONFIGURATION_MODEL_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)


class Configuration:
    @staticmethod
    def new(trial: optuna.trial.BaseTrial, search_space: SearchSpaceModel):

        return Configuration(trial, data=search_space.sample(trial, full=True))

    @staticmethod
    def new_best(trial: optuna.trial.BaseTrial, search_space: SearchSpaceModel):
        hierarchy = search_space.hierarchy()
        params = trial.params

        for manifest_name, tunables in hierarchy.items():
            for key, value in params:
                if key in tunables:
                    tunables[key] = value

        return Configuration(trial, data=hierarchy)

    @staticmethod
    def running_config(search_space: SearchSpaceModel, score: float = 0) -> DefaultConfiguration:
        current_config = {}
        params = {}
        for manifest in search_space.manifests:
            name = manifest.name
            current_config[name] = manifest.get_current_config()
            params.update(current_config[name])
            logger.debug(f'getting manifest {name}:{current_config[name]}')

        distributions = search_space.distributions()

        def make_params_valid(params: dict, distributions: dict) -> dict:
            new_params = {}
            for key, value in params.items():
                new_value = value
                distribution: optuna.distributions.BaseDistribution = distributions.get(key, None)
                if isinstance(distribution, optuna.distributions.IntUniformDistribution) or \
                        isinstance(distribution, optuna.distributions.UniformDistribution):
                    if not (distribution.low <= value <= distribution.high):

                        # use the closest value in the valid distribution as defaul parameter
                        if abs(value - distribution.low) < abs(value - distribution.high):
                            new_value = distribution.low
                        else:
                            new_value = distribution.high
                        logger.warning(f'rescaling {key}:{value} to {new_value}')

                elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                    if value not in distribution.choices:
                        new_value = distribution.choices[0]
                        logger.warning(f'rescaling {key}:{value} to {new_value}')

                elif distribution is None:
                    logger.warning(f'doesn\'t exist a distribution for the parameter {key}:{value}')
                    continue

                new_params[key] = new_value
            return new_params

        trial = optuna.create_trial(
            params=make_params_valid(params, distributions),
            distributions=distributions,
            value=score,
        )

        configuration = DefaultConfiguration(trial, data=search_space.hierarchy(params))
        configuration.score = score

        configuration.trial.number = 0

        return configuration

    __empty_config = None

    @staticmethod
    def empty_config():
        if not Configuration.__empty_config:
            Configuration.__empty_config = EmptyConfiguration()
        return Configuration.__empty_config

    @staticmethod
    def last_confignew(trial: optuna.trial.BaseTrial, search_space: SearchSpaceModel):
        return LastConfiguration(Configuration.new(trial, search_space))

    def __init__(self, trial: optuna.trial.BaseTrial, data: dict):
        self.__trial = trial
        self.__data = data
        self.__name = hashlib.md5(bytes(str(self.data.items()), 'ascii')).hexdigest()
        self.__stats: RunningStats = RunningStats()

    def debug_stats(self) -> list[float]:
        return self.__stats.debug()

    def __lt__(self, other: Configuration):
        return self.median < other.median

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return str(self.serialize())

    def __repr__(self):
        return self.__str__()

    def serialize(self) -> dict:
        return {
            'uid': self.uid,
            'name': self.name,
            'score': self.score,
            'data': self.data,
            'stats': self.__stats.serialize(),
        }

    @property
    def name(self):
        return self.__name

    @property
    def uid(self):
        return self.trial.number

    @property
    def data(self):
        return self.__data

    @property
    def trial(self) -> optuna.Trial:
        return self.__trial

    @property
    def score(self) -> float:
        return self.__stats.curr()

    @score.setter
    def score(self, value: float):
        self.__stats.push(value)
        self.trial.value = self.median

    @property
    def mean(self) -> float:
        return self.__stats.mean()

    @property
    def stddev(self) -> float:
        return self.__stats.standard_deviation()

    @property
    def median(self) -> float:
        return self.__stats.median()

    def final_score(self) -> float:
        return self.median


class DefaultConfiguration(Configuration):
    pass


class EmptyConfiguration(Configuration):
    def __init__(self):
        super(EmptyConfiguration, self).__init__(trial=optuna.trial.FixedTrial(params={}), data={})

    @property
    def score(self) -> float:
        return float('inf')

    @score.setter
    def score(self, value):
        logger.error('cannot update score into a EmptyConfiguration')
        # raise NotImplementedError


class LastConfiguration(Configuration):
    def __init__(self, configuration: Configuration):
        super(LastConfiguration, self).__init__(trial=configuration.trial, data=configuration.data)
