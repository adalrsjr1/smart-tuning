import logging
import os
import sys
import time
import datetime

import kubernetes
from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient

STARTUP_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()


def print_config(toPrint=False):
    if toPrint:
        print(f'\n *** loading config {STARTUP_TIME} ***\n')
        for item in globals().items():
            if item[0].isupper():
                print('\t', item)
        print('\n *** config loaded *** \n')


# K8S_HOST = 'trxrhel7perf-1'
# LOCALHOST = '9.26.100.254'

K8S_HOST = 'trinity01'
LOCALHOST = '127.0.0.1'

# K8S_HOST = 'localhost'
# LOCALHOST = 'localhost'

# K8S_CONF = f'{os.environ.get("HOME")}/.kube/trxrhel7perf-1/config'
K8S_CONF = f'{os.environ.get("HOME")}/.kube/trinity01/config'


def init_k8s(hostname=K8S_HOST):
    if 'KUBERNETES_SERVICE_HOST' in os.environ:
        logging.info('loading K8S configuration')
        kubernetes.config.load_incluster_config()
    else:
        if 'localhost' != hostname:
            logging.info(f'loading remote:{hostname} configuration')
            kubernetes.config.load_kube_config(config_file=K8S_CONF)
        else:
            logging.info('loading localhost configuration')
            kubernetes.config.load_kube_config()


## to disable loggers
FORMAT = '%(asctime)-15s - %(name)-30s %(levelname)-7s - %(threadName)-30s: %(message)s'
INJECTOR_LOGGER = 'injector.smarttuning.ibm'
PLANNER_LOGGER = 'planner.smarttuning.ibm'
CONFIGURATION_MODEL_LOGGER = 'configuration.smarttuning.ibm'
SMARTTUNING_TRIALS_LOGGER = 'sttrials.smarttuning.ibm'
# logging.getLogger('INJECTOR_LOGGER').addHandler(logging.NullHandler())
# logging.getLogger('INJECTOR_LOGGER').propagate = False
#
# logging.getLogger('kubernetes.client.rest').addHandler(logging.NullHandler())
# logging.getLogger('kubernetes.client.rest').propagate = False
#
SAMPLER_LOGGER = 'sampler.smarttuning.ibm'
# logging.getLogger(SAMPLER_LOGGER).addHandler(logging.NullHandler())
# logging.getLogger(SAMPLER_LOGGER).propagate = False
#
APP_LOGGER = 'app.smarttuning.ibm'
# logging.getLogger(APP_LOGGER).addHandler(logging.NullHandler())
# logging.getLogger(APP_LOGGER).propagate = False
KMEANS_LOGGER = 'kmeans.smarttuning.ibm'
BAYESIAN_LOGGER = 'bayesian.smarttuning.ibm'
SEARCH_SPACE_LOGGER = 'searchspace.smarttuning.ibm'
EVENT_LOOP_LOGGER = 'eventloop.smarttuning.ibm'

# debug config
MOCK_WORKLOADS = os.environ.get('MOCK_WORKLOADS', default='jsf,jsp,browsing-jsp,trading-jsp').split(',')
FAIL_FAST = eval(os.environ.get('FAIL_FAST', default='False'))
MOCK = eval(os.environ.get('MOCK', default='True'))
PRINT_CONFIG = eval(os.environ.get('PRINT_CONFIG', default='True'))
LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', default='NOTSET').upper()
logging.basicConfig(level=logging.getLevelName(LOGGING_LEVEL), format=FORMAT)
# proxy config
PROXY_PORT = int(os.environ.get('PROXY_PORT', default=80))
METRICS_PORT = int(os.environ.get('METRICS_PORT', default=9090))
PROXY_NAME = os.environ.get('PROXY_NAME', default='proxy')
PROXY_TAG = os.environ.get('PROXY_TAG', default='smarttuning')  # this should be the same name as in prometheus config
PROXY_IMAGE = os.environ.get('PROXY_IMAGE', default='smarttuning/proxy')
PROXY_CONFIG_MAP = os.environ.get('PROXY_CONFIG_MAP', default='smarttuning-proxy-config')

# mongo config
MONGO_ADDR = os.environ.get('MONGO_ADDR', default=LOCALHOST)
MONGO_PORT = int(os.environ.get('MONGO_PORT', default='30027'))
MONGO_DB = os.environ.get('MONGO_DB', default='smarttuning')
# prometheus config
PROMETHEUS_ADDR = os.environ.get('PROMETHEUS_ADDR', default=LOCALHOST)
PROMETHEUS_PORT = os.environ.get('PROMETHEUS_PORT', default='30099')
SAMPLING_METRICS_TIMEOUT = int(os.environ.get('SAMPLING_METRICS_TIMEOUT', default=5))

# classification config
K = int(os.environ.get('K', default='3'))
DISTANCE_METHOD = os.environ.get('DISTANCE_METHOD', default='hellinger')
URL_SIMILARITY_THRESHOLD = float(os.environ.get('URL_SIMILARITY_THRESHOLD', default='0.1'))

# optimization config
BAYESIAN = eval(os.environ.get('OPTIMIZATION_METHOD', default='True'))
# n_startup_jobs: # of jobs doing random search at begining of optimization
N_STARTUP_JOBS = int(os.environ.get('N_STARTUP_JOBS', default=10))
# n_EI_candidades: number of config samples draw before select the best. lower number encourages exploration
N_EI_CANDIDATES = int(os.environ.get('N_EI_CANDIDATES', default=24))
# gamma: p(y) in p(y|x) = p(x|y) * p(x)/p(y) or specifically  1/(gamma + g(x)/l(x)(1-gamma))
GAMMA = float(os.environ.get('GAMMA', default=0.25))
NUMBER_ITERATIONS = int(float(
    os.environ.get('NUMBER_ITERATIONS', default='10')))
MAX_N_ITERATION_NO_IMPROVEMENT = int(float(
    os.environ.get('MAX_N_ITERATION_NO_IMPROVEMENT', default='15')))
ITERATIONS_BEFORE_REINFORCE = int(os.environ.get('ITERATIONS_BEFORE_REINFORCE', default='3'))
RESTART_TRIGGER = float(os.environ.get('RESTART_TRIGGER', default='1'))
TRY_BEST_AT_EVERY = int(os.environ.get('TRY_BEST_AT_EVERY', default=ITERATIONS_BEFORE_REINFORCE))
REINFORCEMENT_RATIO = float(os.environ.get('REINFORCEMENT_RATIO', default='1.0'))
METRIC_THRESHOLD = float(os.environ.get('METRIC_THRESHOLD', default='0.2'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', default=time.time()))
# OBJECTIVE = compile(os.environ.get('OBJECTIVE', default='-throughput/(memory / 2 ** 20)'), '<string>', 'eval')
OBJECTIVE = str(os.environ.get('OBJECTIVE', default='memory'))
# sampling config
SAMPLE_SIZE = float(os.environ.get('SAMPLE_SIZE', default='0.3334'))
WAITING_TIME = int(os.environ.get('WAITING_TIME', default='10'))
GATEWAY_NAME = os.environ.get('GATEWAY_NAME', default='acmeair-nginxservice')

# failfast threshold
THROUGHPUT_THRESHOLD = float(os.environ.get('THROUGHPUT_THRESHOLD', default='1.0'))

QUANTILE = float(os.environ.get('QUANTILE', default='1.0'))

# actuator config
NAMESPACE = os.environ.get('NAMESPACE', 'default')

# # deprecated -- to remove
# SEARCH_SPACE_NAME = os.environ.get('SEARCH_SPACE_NAME', default='default')
# NAMESPACE_PROD = os.environ.get('NAMESPACE_PROD', 'default')
# SEARCHSPACE_PATH = os.environ.get('SEARCHSPACE_PATH', default='')
# CONFIGMAP_NAME = os.environ.get('CONFIGMAP_NAME', default='tuning-config')
# CONFIGMAP_PROD_NAME = os.environ.get('CONFIGMAP_PROD_NAME', default='tuning-config')
# POD_REGEX = os.environ.get(''
#                            'POD_REGEX', default='acmeair-.+servicessmarttuning-.+')
# POD_PROD_REGEX = os.environ.get('POD_PROD_REGEX', default='acmeair-.+services-.+')


print_config(PRINT_CONFIG)
_executor = None
_client = None

__customApi = None


def customApi() -> kubernetes.client.CustomObjectsApi:
    global __customApi
    if not __customApi:
        __customApi = kubernetes.client.CustomObjectsApi()
    return __customApi


__coreV1Api = None


def coreApi() -> kubernetes.client.CoreV1Api:
    global __coreV1Api
    if not __coreV1Api:
        __coreV1Api = kubernetes.client.CoreV1Api()
    return __coreV1Api


__appsApi = None


def appsApi() -> kubernetes.client.AppsV1Api:
    global __appsApi
    if not __appsApi:
        __appsApi = kubernetes.client.AppsV1Api()
    return __appsApi


def executor() -> ThreadPoolExecutor:
    global _executor
    if not _executor:
        _executor = ThreadPoolExecutor()
    return _executor


def mongo() -> MongoClient:
    global _client
    if not _client:
        _client = MongoClient(MONGO_ADDR, MONGO_PORT)
    return _client


def ping(address: str, port: int) -> bool:
    import socket
    a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    location = (address, port)

    result_of_check = a_socket.connect_ex(location)
    a_socket.close()
    return result_of_check == 0


import ctypes


def terminate_thread(thread):
    """Terminates a python thread from another thread.

    :param thread: a threading.Thread instance
    """
    if not thread.is_alive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if 0 == res:
        raise ValueError("nonexistent thread id")
    elif 1 <= res:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def shutdown():
    mongo().close()
    executor().shutdown(wait=False)
    for t in executor()._threads:
        try:
            terminate_thread(t)
        except SystemError:
            logging.exception('error while shutdown thread pool')

# if __name__ == '__main__':
# init_k8s(K8S_HOST)
# podList = coreApi().list_pod_for_all_namespaces()
# for pod in podList.items:
#     print(pod.metadata.name)
# print(ping(PROMETHEUS_ADDR, int(PROMETHEUS_PORT)))
