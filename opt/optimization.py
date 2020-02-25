from hyperopt import plotting, Trials, fmin, tpe, hp
from common.timeutil import now
import numpy as np

import dotenv
import re


class EnvConfig:

    def __init__(self, path, types):
        self.path = path
        self.types = types

    def __set_with_type(self, key, value):
        if key in self.types.keys():
            if type(value) is str and self.__isnumber(value):
                value = self.__set_with_type(key, self.__str_to_number(value))
            return str(self.types[key](value))

        return str(value)

    def set_value(self, key, value):
        value_to_set = self.__set_with_type(key, value)
        dotenv.set_key(dotenv_path=self.path, key_to_set=key, value_to_set=value_to_set, quote_mode='never')

    def get_value(self, key):
        return self.__sanitize_value(key, dotenv.get_key(dotenv_path=self.path, key_to_get=key))

    def __isnumber(self, value):
        try:
            self.__str_to_number(value)
            return True
        except:
            return False

    def __str_to_number(self, value):
        pattern = re.compile('^\d+[\.$\d*]*')
        match = pattern.match(value)
        return match.group()

    def __sanitize_value(self, key, value):
        if key in self.types.keys():
            if self.types[key] != str:
                numeric_value = self.__str_to_number(value)
                if numeric_value != None:
                    return self.types[key](float(numeric_value))

        return str(value)

    def keys(self):
        return [key for key in dotenv.dotenv_values(dotenv_path=self.path).keys()]

    def values(self):
        return [value for value in dotenv.dotenv_values(dotenv_path=self.path).values()]

    def items(self):
        return [item for item in dotenv.dotenv_values(dotenv_path=self.path).items()]


class SearchSpace:
    def __init__(self, domain=None):
        if not domain:
            domain = {}
        self.domain = domain

    def add_to_domain(self, key, lower, upper, type):
        """
        type: str, int, float
        """
        self.domain.update({key: (lower, upper, type)})

    def rem_from_domain(self, key):
        del self.domain[key]

    def dimension(self, key):
        lower = self.domain[key][0]
        upper = self.domain[key][1]
        type = self.domain[key][2]

        dimension = hp.uniform(key, lower, upper)
        if type == int:
            dimension = hp.quniform(key, lower, upper, 1)

        return {key: dimension}

    def search_space(self):
        space = {}
        for key, value in self.domain.items():
            space.update(self.dimension(key))
        return space


class Optimization:
    def __init__(self, trial, seed, space, objective, max_evals):
        self.space = space
        self.objective = objective
        self.max_evals = max_evals
        self.trials = trial
        self.seed = seed

    def optimize(self):
        if not self.trials:
            self.trials = Trials()

        result = {'start': now()}
        best = fmin(fn=self.objective.f,
                    trials=self.trials,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.max_evals,
                    rstate=np.random.RandomState(self.seed))
        result.update({'end': now(), 'tuning': best})
        best_loss = min(self.trials.losses())
        result.update({'best': best_loss})
        # save this result to mongo
        return result

    def plot_vars(self):
        plotting.main_plot_vars(self.trials, colorize_best=True, arrange_by_loss=True)


class ObjectiveFunction:

    def __init__(self, env_params, function, save=False, filepath='.'):
        self.env_params = env_params
        self.function = function
        self.save = save
        self.filepath = filepath

    def f(self, params):
        [self.env_params.set_value(key, value) for key, value in params.items()]
        print(params)
        metric = self.function.execute(params)

        if self.save:
            with open(f'{self.filepath}/debug.output','w+') as file:
                file.write(f'{params} loss:{metric}\n')

        print(f'{params} loss:{metric}')
        return metric

    def run_application(self):
        self.application.execute(self.env_params)

    def stop_application(self):
        self.application.stop()
