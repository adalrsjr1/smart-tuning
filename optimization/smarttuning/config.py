import logging
import os
import time
import kubernetes
from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient


def print_config(toPrint=False):
    if toPrint:
        print('\n *** loading config ***\n')
        for item in globals().items():
            if item[0].isupper():
                print('\t', item)
        print('\n *** config loaded *** \n')

def init_k8s(hostname='localhost'):
    if 'KUBERNETES_SERVICE_HOST' in os.environ:
        logging.info('loading K8S configuration')
        kubernetes.config.load_incluster_config()
    else:
        if 'trxrhel7perf-1' == hostname:
            logging.info('loading trxrhel7perf-1 configuration')
            kubernetes.config.load_kube_config(config_file=K8S_CONF)
        else:
            logging.info('loading localhost configuration')
            kubernetes.config.load_kube_config()

K8S_HOST = 'trxrhel7perf-1'
K8S_CONF = '/Users/adalbertoibm.com/.kube/trxrhel7perf-1/config'
LOCALHOST = '9.26.100.254'
## to disable loggers
FORMAT = '%(asctime)-15s - %(name)-30s %(levelname)-7s - %(threadName)-30s: %(message)s'
INJECTOR_LOGGER = 'injector.smarttuning.ibm'
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
MOCK = eval(os.environ.get('MOCK', default='True'))
PRINT_CONFIG = eval(os.environ.get('PRINT_CONFIG', default='False'))
LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', default='DEBUG').upper()
logging.basicConfig(level=logging.getLevelName(LOGGING_LEVEL), format=FORMAT)
# proxy config
PROXY_PORT = int(os.environ.get('PROXY_PORT', default=80))
METRICS_PORT = int(os.environ.get('METRICS_PORT', default=9090))
PROXY_NAME = os.environ.get('PROXY_NAME', default='proxy')
PROXY_TAG = os.environ.get('PROXY_TAG', default='smarttuning')  # this should be the same name as in prometheus config
PROXY_IMAGE = os.environ.get('PROXY_IMAGE', default='smarttuning/proxy:latest')
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
SEARCH_SPACE_NAME = os.environ.get('SEARCH_SPACE_NAME', default='default')
BAYESIAN = eval(os.environ.get('OPTIMIZATION_METHOD', default='True'))
N_STARTUP_JOBS = int(os.environ.get('N_STARTUP_JOBS', default=20))
N_EI_CANDIDATES = int(os.environ.get('N_EI_CANDIDATES', default=24))
GAMMA = float(os.environ.get('GAMMA', default=0.25))
NUMBER_ITERATIONS = int(os.environ.get('NUMBER_ITERATIONS', default='3'))
METRIC_THRESHOLD = float(os.environ.get('METRIC_THRESHOLD', default='0.2'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', default=time.time()))
OBJECTIVE = compile(os.environ.get('OBJECTIVE', default='throughput / memory'), '<string>', 'eval')
# sampling config
SAMPLE_SIZE = float(os.environ.get('SAMPLE_SIZE', default='0.3334'))
WAITING_TIME = int(os.environ.get('WAITING_TIME', default='300'))
POD_REGEX = os.environ.get('POD_REGEX', default='acmeair-.+servicessmarttuning-.+')
POD_PROD_REGEX = os.environ.get('POD_PROD_REGEX', default='acmeair-.+services-.+')
QUANTILE = float(os.environ.get('QUANTILE', default='1.0'))

# actuator config
CONFIGMAP_NAME = os.environ.get('CONFIGMAP_NAME', default='tuning-config')
CONFIGMAP_PROD_NAME = os.environ.get('CONFIGMAP_PROD_NAME', default='tuning-config')
NAMESPACE = os.environ.get('NAMESPACE', 'default')

# deprecated -- to remove
NAMESPACE_PROD = os.environ.get('NAMESPACE_PROD', 'default')
SEARCHSPACE_PATH = os.environ.get('SEARCHSPACE_PATH', default='')


print_config(PRINT_CONFIG)
_executor = None
_client = None


def executor():
    global _executor
    if not _executor:
        _executor = ThreadPoolExecutor()
    return _executor


def mongo():
    global _client
    if not _client:
        _client = MongoClient(MONGO_ADDR, MONGO_PORT)
    return _client

def ping(address:str, port:int) -> bool:
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

if __name__ == '__main__':
    print(ping(PROMETHEUS_ADDR, int(PROMETHEUS_PORT)))
