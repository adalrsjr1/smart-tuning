from hyperopt import fmin, tpe, hp, space_eval, Trials


class HyperOpt:
    def __init__(self, objective, search_space):
        self.objective = objective
        self.search_space = search_space

    def optimize(self, max_evals):
        trials = Trials()
        best = fmin(self.objective, self.search_space, trials=trials, algo=tpe.suggest, max_evals=max_evals)
        return trials.best_trial, best


if __name__ == '__main__':
    # define an objective function
    import numpy as np


    def objective(args):
        t = args['x0'] + args['x1'] + args['x2'] + args['x3'] + args['x4']
        return np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * 2.25 * 2 * np.pi * t)


    # define a search space
    space = {'x0': hp.uniform('x0', -1.0, 1.0),
             'x1': hp.uniform('x1', -0.1, 1.1),
             'x2': hp.uniform('x2', -0.2, 1.2),
             'x3': hp.uniform('x3', -0.3, 1.3),
             'x4': hp.uniform('x4', -0.4, 1.4)}

    opt = HyperOpt(objective, space)
    best_trial = opt.optimize(100)

    print(best_trial)
