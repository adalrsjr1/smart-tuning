import configsampler as cs
import json
from queue import Queue
import numpy as np
from hyperopt import fmin, tpe, rand
import config

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED as FUTURE_ALL_COMPLETED


def load_search_space(config_path)-> cs.SearchSpace:
    search_space = cs.SearchSpace({})

    print('\nloading search space')
    with open(config_path) as json_file:
        data = json.load(json_file)
        for item in data:
            print('\t', item)
            search_space.add_to_domain(
                key=item.get('key', None),
                lower=item.get('lower', None),
                upper=item.get('upper', None),
                options=item.get('options', None),
                type=item.get('type', None)
            )

    return search_space

space = load_search_space('search_space.json')
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
        print(e)
    return metric

def sample():
    parameters = chn_out.get(True)
    # print('[2] <<< ', parameters)
    metric = np.random.normal(loc=0, scale=0.01)
    # time.sleep(5)
    chn_in.put(metric)
    # print('[3] >>> ', metric)
    return parameters, metric

surrogate = rand.suggest
if config.BAYESIAN:
    surrogate = tpe.suggest

config.executor.submit(fmin, fn=objective, space=space.search_space(), algo=surrogate, max_evals=int(1e15), verbose=False, show_progressbar=False)
# done = wait([o], timeout=None, return_when=FUTURE_ALL_COMPLETED)

