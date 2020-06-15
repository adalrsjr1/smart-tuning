from queue import Queue
import numpy as np
import traceback
from hyperopt import fmin, tpe, rand, Trials, STATUS_OK
import pickle
import time
import config

# space = load_search_space('search_space.json')
space = None
chn_in = Queue(maxsize=1)
chn_out = Queue(maxsize=1)
trials = Trials()
running = False

def objective(params):
    try:
        # search space
        # time.sleep(1)
        # print('[1] >>> ', params)
        chn_out.put(params)
        metric = chn_in.get(True)
    except Exception as e:
        traceback.print_exc()
        print(e)
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
            # n_EI_candidades: number of config samples draw before select the best
            # gamma: p(y) in p(y|x) = p(x|y) * p(x)/p(y) or specifically  1/(gamma + g(x)/l(x)(1-gamma))
            surrogate = partial(tpe.suggest, n_startup_jobs=20, n_EI_candidates=24, gamma=0.25)

        config.executor.submit(fmin, fn=objective, trials=trials, space=space, algo=surrogate, max_evals=int(1e15), verbose=False, show_progressbar=False, rstate= np.random.RandomState(config.RANDOM_SEED))
    # done = wait([o], timeout=None, return_when=FUTURE_ALL_COMPLETED)
        running = True

