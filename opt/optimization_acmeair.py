from hyperopt import Trials
from opt.optimization import EnvConfig, SearchSpace, ObjectiveFunction, Optimization
from common.dataaccess import MongoAccessLayer, PrometheusAccessLayer
from common.timeutil import now, second

import os
import sys
import argparse


class AcmeAirOptimization:
    def __init__(self, metric_handler, time_interval, app_url, app_port, app_path):
        self.DOCKER_COMPOSE = 'resources/docker-compose.yml'
        self.WORKLOAD = 'workload-'
        self.METRIC = metric_handler
        self.time_interval = time_interval
        self.app_url = app_url
        self.app_port = app_port
        self.app_path = app_path

    def execute(self, params):
        best = 0
        if self.run_app() == 0:
            if self.health(self.app_url, self.app_port, self.app_path) == 0:
                if self.run_client(self.app_url, self.app_port) == 0:
                    best = -self.METRIC.throughput(now(past=second(self.time_interval)), now(), self.time_interval)

        return best

    def run_app(self):
        return os.system(f'docker-compose -f {self.DOCKER_COMPOSE} up -d acmeair')

    def run_client(self, url, port):
        clients = 100
        rampup = 10
        import random
        return os.system(
            f'resources/run_jmeter.sh {self.WORKLOAD}{random.randint(1, 5)} 1 {clients} {self.time_interval} {rampup} {url} {port}')

    def stop(self):
        return os.system(f'docker-compose -f {self.DOCKER_COMPOSE} down acmeair')

    def health(self, app_url, app_port, app_path):
        print(f'resources/health.sh -a http://{app_url}:{app_port}/{app_path}')
        return os.system(f'resources/health.sh -a http://{app_url}:{app_port}/{app_path}')


class AcmeAirMetricHandler:
    def __init__(self, prometheus_client):
        self.client = prometheus_client

    def throughput(self, start, end, step, to_round=False):
        response = self.client.avg('vendor_servlet_request_total{servlet="acmeair_webapp_com_acmeair_web_AcmeAirApp"}',
                                   start, end, step)
        return response.data[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prometheus-url', type=str, default='localhost')
    parser.add_argument('--prometheus-port', type=int, default=9090)
    parser.add_argument('--mongo-url', type=str, default='localhost')
    parser.add_argument('--mongo-port', type=int, default=27017)
    parser.add_argument('--mongo-db', type=str, default='acmeair_db_experiments')
    parser.add_argument('--mongo-collection', type=str, default='acmeair_collection_tuning')

    parser.add_argument('--application-url', type=str, default='localhost')
    parser.add_argument('--application-port', type=str, default='9092')
    parser.add_argument('--application-path', type=str, default='acmeair-webapp')

    parser.add_argument('--n-iterations', type=int, default=1)

    args = parser.parse_args()

    client = PrometheusAccessLayer(args.prometheus_url, args.prometheus_port)
    handler = AcmeAirMetricHandler(client)

    mongo = MongoAccessLayer(args.mongo_url, args.mongo_port, args.mongo_db)
    mongo_collections = mongo.collection(args.mongo_collection)

    env_config = EnvConfig('resources/docker-compose.env',
                           {'CORE_THREADS': int,
                            'MAX_KEEP_ALIVE_REQUESTS': int,
                            'PERSIST_TIMEOUT': int,
                            'MAX_THREADS': int})

    search_space = SearchSpace({})

    search_space.add_to_domain('CORE_THREADS', 4, 20, int)
    search_space.add_to_domain('MAX_KEEP_ALIVE_REQUESTS', 1, 200, int)
    search_space.add_to_domain('PERSIST_TIMEOUT', 1, 30, int)
    search_space.add_to_domain('MAX_THREADS', 4, 20, int)

    acmeair_optimization = AcmeAirOptimization(handler, 60, args.application_url, args.application_port,
                                               args.application_path)
    objective = ObjectiveFunction(env_config, acmeair_optimization, save=True, filepath='.')

    optimization = Optimization(Trials(), 2020, search_space.search_space(), objective, args.n_iterations)

    _start = now()
    optimal = optimization.optimize()
    _end = now()
    result = {'start': _start, 'end': _end}
    result.update(optimal)
    mongo.store(result, mongo_collections)
    print(result)

    for key, value in optimal.items():
        env_config.set_value(key, value)

    acmeair_optimization.run_app()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
