import os
import time
import logging
# import kubernetes as k8s
from concurrent.futures import ThreadPoolExecutor, wait as ThreadWait, ALL_COMPLETED as FUTURE_ALL_COMPLETED

from pymongo import MongoClient

def print_config(toPrint=False):
    if toPrint:
        print('\n *** loading config ***\n')
        for item in globals().items():
            if item[0].isupper():
                print('\t', item)
        print('\n *** config loaded *** \n')

## to disable loggers
INJECTOR_LOGGER = 'injector.smarttuning.ibm'
# logging.getLogger('INJECTOR_LOGGER').addHandler(logging.NullHandler())
# logging.getLogger('INJECTOR_LOGGER').propagate = False
#
# logging.getLogger('kubernetes.client.rest').addHandler(logging.NullHandler())
# logging.getLogger('kubernetes.client.rest').propagate = False
#
SAMPLER_LOGGER = 'sapler.smarttuning.ibm'
# logging.getLogger(SAMPLER_LOGGER).addHandler(logging.NullHandler())
# logging.getLogger(SAMPLER_LOGGER).propagate = False
#
APP_LOGGER = 'app.smarttuning.ibm'
# logging.getLogger(APP_LOGGER).addHandler(logging.NullHandler())
# logging.getLogger(APP_LOGGER).propagate = False
KMEANS_LOGGER = 'kmeans.smarttuning.ibm'
BAYESIAN_LOGGER = 'bayesian.smarttuning.ibm'
SEARCH_SPACE_LOGGER = 'searchspace.smarttuning.ibm'

# debug config
MOCK = eval(os.environ.get('MOCK', default='True'))
PRINT_CONFIG = eval(os.environ.get('PRINT_CONFIG', default='False'))
LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', default='DEBUG').upper()
logging.basicConfig(level=logging.getLevelName(LOGGING_LEVEL), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# proxy config
PROXY_PORT = int(os.environ.get('PROXY_PORT', default=80))
METRICS_PORT = int(os.environ.get('METRICS_PORT', default=9090))
PROXY_NAME = os.environ.get('PROXY_NAME', default='proxy')
PROXY_TAG = os.environ.get('PROXY_TAG', default='smarttuning') # this should be the same name as in prometheus config
PROXY_IMAGE = os.environ.get('PROXY_IMAGE', default='smarttuning/proxy')
PROXY_CONFIG_MAP = os.environ.get('PROXY_CONFIG_MAP', default='smarttuning-proxy-config')

# mongo config
MONGO_ADDR = os.environ.get('MONGO_ADDR', default='127.0.0.1')
MONGO_PORT = int(os.environ.get('MONGO_PORT', default='30027'))
MONGO_DB = os.environ.get('MONGO_DB', default='smarttuning')

# prometheus config
PROMETHEUS_ADDR = os.environ.get('PROMETHEUS_ADDR', default='localhost')
PROMETHEUS_PORT = os.environ.get('PROMETHEUS_PORT', default='30090')
SAMPLING_METRICS_TIMEOUT = int(os.environ.get('SAMPLING_METRICS_TIMEOUT', default=15))

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
OBJECTIVE = compile(os.environ.get('OBJECTIVE', default='throughput/memory'),'<string>', 'eval')
# sampling config
SAMPLE_SIZE = float(os.environ.get('SAMPLE_SIZE', default='1.0'))
WAITING_TIME = int(os.environ.get('WAITING_TIME', default='2'))
POD_REGEX = os.environ.get('POD_REGEX', default='.*tuning.*')
POD_PROD_REGEX = os.environ.get('POD_PROD_REGEX', default='.*tuningprod.*')
QUANTILE = float(os.environ.get('QUANTILE', default='1.0'))

# actuator config
CONFIGMAP_NAME = os.environ.get('CONFIGMAP_NAME', default='tuning-config')
CONFIGMAP_PROD_NAME = os.environ.get('CONFIGMAP_PROD_NAME', default='tuning-config')
NAMESPACE = os.environ.get('NAMESPACE', 'default')

# deprecated -- to remove
NAMESPACE_PROD = os.environ.get('NAMESPACE_PROD', 'default')
SEARCHSPACE_PATH = os.environ.get('SEARCHSPACE_PATH',default='')
# CONFIG_PATH = os.environ.get('CONFIG_PATH', default='/etc')
# REGISTER_SERVER_PORT = int(os.environ.get('REGISTER_SERVER_PORT', default='5000'))
# REGISTER_SERVER_ADDR = os.environ.get('REGISTER_SERVER_ADDR', default='0.0.0.0')
# SYNC_PORT = int(os.environ.get('SYNC_PORT', default='5000'))

print_config(PRINT_CONFIG)


executor = ThreadPoolExecutor()
client = MongoClient(MONGO_ADDR, MONGO_PORT)

def shutdown():
    client.close()
    executor.shutdown()


