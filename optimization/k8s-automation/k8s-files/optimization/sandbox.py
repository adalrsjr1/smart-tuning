import hyperopt
from hyperopt import hp

space = {
    'a': hp.quniform('a', 10, 100, 10),
    'b': hp.quniform('b', 1.0, 10.0, -1)

}

for _ in range(10):
    print(hyperopt.pyll.stochastic.do_sample(space))