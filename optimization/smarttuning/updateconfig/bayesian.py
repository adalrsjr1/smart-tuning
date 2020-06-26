from queue import Queue
import numpy as np
from hyperopt import fmin, tpe, rand, Trials, STATUS_OK, STATUS_FAIL, space_eval
import logging
import time
import config
from sampler import Metric

logger = logging.getLogger(config.BAYESIAN_LOGGER)
logger.setLevel(logging.DEBUG)

class BayesianDTO:
    def __init__(self, metric=Metric.zero(), classification=''):
        self.metric = metric
        self.classification = classification

class BayesianEngine:
    ## to make search-space dynamic
    ## https://github.com/hyperopt/hyperopt/blob/2814a9e047904f11d29a8d01e9f620a97c8a4e37/tutorial/Partial-sampling%20in%20hyperopt.ipynb
    def __init__(self, id:str, space=None, is_bayesian=True, max_evals=int(1e15)):
        self._id = id
        self._running = False
        self._trials = Trials()
        self._space = space

        surrogate = rand.suggest
        if is_bayesian:
            from functools import partial
            # n_startup_jobs: # of jobs doing random search at begining of optimization
            # n_EI_candidades: number of config samples draw before select the best. lower number encourages exploration
            # gamma: p(y) in p(y|x) = p(x|y) * p(x)/p(y) or specifically  1/(gamma + g(x)/l(x)(1-gamma))
            surrogate = partial(tpe.suggest, n_startup_jobs=config.N_STARTUP_JOBS,
                                n_EI_candidates=config.N_EI_CANDIDATES, gamma=config.GAMMA)

        # add early_stop when next Hyperopt version came out
        # https://github.com/hyperopt/hyperopt/blob/abf6718951eecc1c43d591f59da321f2de5a8cbf/hyperopt/tests/test_fmin.py#L336
        config.executor.submit(fmin, fn=self.objective, trials=self.trials(), space=self._space, algo=surrogate, max_evals=max_evals,
                               verbose=False, show_progressbar=False,
                               rstate=np.random.RandomState(config.RANDOM_SEED))
        self._running = True

    def id(self)->str:
        return self._id

    def trials(self):
        return self._trials

    def is_running(self):
        return self._running

    def objective(self, params):
        # follow this hint for implement multiple workload types
        # https://github.com/hyperopt/hyperopt/issues/181
        status = STATUS_FAIL
        loss = float('inf')
        classification = ''
        try:
            chn_out.put(params)
            dto: BayesianDTO = chn_in.get(True)
            loss = dto.metric.objective()
            classification = dto.classification
            status = STATUS_OK
        except Exception:
            logger.exception('at bayesian core')
        finally:
            return {
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


    def get(self):
        parameters = chn_out.get(True)
        return parameters


    def put(self, dto:BayesianDTO):
        chn_in.put(dto)


    def best_so_far(self):
        min = self.trials().argmin
        return space_eval(self._space, min)


    def sample(self, metric):
        parameters = chn_out.get(True)
        chn_in.put(metric)
        return parameters, metric


# space = load_search_space('search_space.json')
space = None
chn_in = Queue(maxsize=1)
chn_out = Queue(maxsize=1)
trials = Trials()
running = False


def objective(params):
    # follow this hint for implement multiple workload types
    # https://github.com/hyperopt/hyperopt/issues/181
    try:
        # search space
        chn_out.put(params)
        metric = chn_in.get(True)
    except Exception as e:
        logger.exception('at bayesian core')
    return {
        'loss': metric.objective(),
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        # 'classification': classficiation.id,
        # # -- attachments are handled differently
        # https://github.com/hyperopt/hyperopt/wiki/FMin
        # 'attachments':
        #     {'classification': pickle.dumps(classficiation)}
    }


def best_trial(key):
    results = trials.results
    min_idx = 0
    min_loss = float('inf')
    for idx, result in enumerate(results):
        if result['loss'] < min_loss:
            min_loss = result['loss']
            min_idx = idx

    return results[min_idx][key]


def sample(metric):
    parameters = chn_out.get(True)
    chn_in.put(metric)
    return parameters, metric


def get():
    parameters = chn_out.get(True)
    return parameters


def put(metric):
    chn_in.put(metric)


def init(search_space):
    global running
    if not running:
        global space
        space = search_space

        surrogate = rand.suggest
        if config.BAYESIAN:
            from functools import partial
            # n_startup_jobs: # of jobs doing random search at begining of optimization
            # n_EI_candidades: number of config samples draw before select the best. lower number encourages exploration
            # gamma: p(y) in p(y|x) = p(x|y) * p(x)/p(y) or specifically  1/(gamma + g(x)/l(x)(1-gamma))
            surrogate = partial(tpe.suggest, n_startup_jobs=20, n_EI_candidates=24, gamma=0.25)

        config.executor.submit(fmin, fn=objective, trials=trials, space=space, algo=surrogate, max_evals=int(1e15),
                               verbose=False, show_progressbar=False, rstate=np.random.RandomState(config.RANDOM_SEED))
        running = True
