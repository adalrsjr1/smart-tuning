import bayesian
import config
if __name__ == '__main__':
    # print(bayesian.sample())
    print(bayesian.get())
    bayesian.put(10)
    # print(bayesian.get())
    config.shutdown()