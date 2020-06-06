from queue import Queue
import numpy as np
import traceback
from hyperopt import fmin, tpe, rand
import config

# space = load_search_space('search_space.json')
space = None
chn_in = Queue(maxsize=1)
chn_out = Queue(maxsize=1)

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
    return metric.objective()

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
    global space
    space = search_space

    surrogate = rand.suggest
    if config.BAYESIAN:
        surrogate = tpe.suggest

    config.executor.submit(fmin, fn=objective, space=space, algo=surrogate, max_evals=int(1e15), verbose=False, show_progressbar=False, rstate= np.random.RandomState(config.RANDOM_SEED))
# done = wait([o], timeout=None, return_when=FUTURE_ALL_COMPLETED)

