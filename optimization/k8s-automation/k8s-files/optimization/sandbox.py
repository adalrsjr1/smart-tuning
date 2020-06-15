from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
import numpy as np
import time
import pickle
from functools import partial
from pprint import pprint
import heapq
# "acmeair-config-train":{"HTTP_MAX_KEEP_ALIVE_REQUESTS":"260","MONGO_CONNECTION_TIMEOUT":"15","MONGO_MAX_CONNECTIONS":"270"},"acmeair-tuning":{"cpu":8.0,"memory":304.0},
# "jvm-config-train":{"container_support":"-XX:+UseContainerSupport","gc":"-Xgcpolicy:optthruput","virtualized":"-Xtune:virtualized"}}

space = {
    'acmeair-config-train': {
        'MONGO_CONNECTION_TIMEOUT': hp.quniform('MONGO_CONNECTION_TIMEOUT', 10, 30, 1),
        'MONGO_MAX_CONNECTIONS': hp.quniform('MONGO_MAX_CONNECTIONS', 4, 300, 10),
        'HTTP_MAX_KEEP_ALIVE_REQUESTS': hp.quniform('HTTP_MAX_KEEP_ALIVE_REQUESTS', 4, 300, 10)
    },
    'acmeair-tuning': {
        'cpu': hp.quniform('cpu', 1, 15, 1),
        'memory': hp.quniform('memory', 256, 1024, 16)
    },
    'jvm-config-train': {
        'container_support': hp.choice('contianer_support', ['-XX:+UseContainerSupport', '-XX:-UseContainerSupport']),
        'gc': hp.choice('gc', ['-Xgcpolicy:gencon', '-Xgc:concurrentScavenge','-Xgcpolicy:metronome','-Xgcpolicy:optavgpause','-Xgcpolicy:optthruput']),
        'virtualized': hp.choice('virtualized', ['-Xtune:virtualized'])
    }

}

def objective(args:dict):
    # status = STATUS_FAIL if np.random.uniform(0, 1) < 0.3 else STATUS_OK
    return {
        'loss': np.random.randint(50, 100),
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
        }

if __name__ == '__main__':
    trials=Trials()
    surrogate = partial(tpe.suggest, n_startup_jobs=4, n_EI_candidates=24, gamma=0.25)
    best = fmin(objective, space, trials=trials, algo=surrogate, max_evals=100, verbose=True, rstate=np.random.RandomState(31))
    print(best)

    # pprint(trials.trials)
    # for t in trials.trials:
    #     print(pickle.loads(trials.trial_attachments(t)['time_module']), type(pickle.loads(trials.trial_attachments(t)['time_module'])))
    #     print(t)
       # print(t['misc']['vals'])
    pprint(trials.results)
    # pprint(trials.losses())

    # for _ in range(30):
    #     print(tpe.pyll.stochastic.sample(space))
