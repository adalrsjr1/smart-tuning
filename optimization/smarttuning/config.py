import datetime
import logging
import logging.config
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import kubernetes
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

# K8S_HOST = 'trinity01'
# LOCALHOST = '127.0.0.1'

K8S_HOST = 'localhost'
LOCALHOST = 'localhost'

# K8S_CONF = f'{os.environ.get("HOME")}/.kube/trxrhel7perf-1/config'
K8S_CONF = f'{os.environ.get("HOME")}/.kube/trinity01/config'

__loaded = False


def init_k8s(hostname=K8S_HOST):
    global __loaded
    if __loaded:
        return

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

    __loaded = True

# loggers names
# FORMAT = '%(asctime)-15s - %(name)-30s %(levelname)-7s - %(threadName)-30s: %(message)s'
FORMAT = '%(asctime)-15s - %(levelname)-7s - %(message)s'
INJECTOR_LOGGER = 'injector.smarttuning.ibm'
PLANNER_LOGGER = 'planner.smarttuning.ibm'
CONFIGURATION_MODEL_LOGGER = 'configuration.smarttuning.ibm'
MODELS_LOGGER = 'models.smarttuning.ibm'
SMARTTUNING_TRIALS_LOGGER = 'sttrials.smarttuning.ibm'
SAMPLER_LOGGER = 'sampler.smarttuning.ibm'
APP_LOGGER = 'app.smarttuning.ibm'
KMEANS_LOGGER = 'kmeans.smarttuning.ibm'
BAYESIAN_LOGGER = 'bayesian.smarttuning.ibm'
SEARCH_SPACE_LOGGER = 'searchspace.smarttuning.ibm'
EVENT_LOOP_LOGGER = 'eventloop.smarttuning.ibm'
WORKLOAD_LOGGER = 'workload.smarttuning.ibm'
METRIC_LOGGER = 'metric.smarttuning.ibm'

# disable k8s loggings
logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True})
# disable optuna warnings
warnings.filterwarnings("ignore", category=Warning)

# debug config
PRINT_CONFIG = eval(os.environ.get('PRINT_CONFIG', default='False'))
# ST_LOG_LEVEL = 60
ST_LOG_LEVEL = logging.NOTSET
logging.addLevelName(ST_LOG_LEVEL, 'SMART_TUNING')
LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', default='smart_tuning').upper()
logging.basicConfig(level=logging.getLevelName(LOGGING_LEVEL), format=FORMAT)
# logging.basicConfig(level=logging.WARNING, format=FORMAT)
logger = logging.getLogger(PLANNER_LOGGER)

# timeout before update workload
SAMPLER_CONFIG = os.environ.get('SAMPLER_CONFIG', default='sampler.json')

# if true, training replica will be behind a dedicated service
TWO_SERVICES = eval(os.environ.get('TWO_SERVICES', default='True'))
WORKLOAD_CLASSIFIER = os.environ.get('WORKLOAD_CLASSIFIER', default='RPS')
JMETER_CFG_WORKLOAD = os.environ.get('JMETER_CFG_WORKLOAD', default='JUSERS')
JMETER_CM = os.environ.get('JMETER_CM', default='jmeter-cm')
WORKLOAD_BANDS = os.environ.get('WORKLOAD_BANDS', default='').split(',')  # no bands
FAIL_FAST = eval(os.environ.get('FAIL_FAST', default='True'))

# proxy config
NO_PROXY = eval(os.environ.get('NO_PROXY', default='False'))
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
ST_METRICS_PORT = int(os.environ.get('ST_METRICS_PORT', default='8000'))
PROMETHEUS_ADDR = os.environ.get('PROMETHEUS_ADDR', default=LOCALHOST)
PROMETHEUS_PORT = os.environ.get('PROMETHEUS_PORT', default='30099')
SAMPLING_METRICS_TIMEOUT = int(os.environ.get('SAMPLING_METRICS_TIMEOUT', default=5))

# optimization config
BAYESIAN = eval(os.environ.get('OPTIMIZATION_METHOD', default='True'))
# n_startup_jobs: # of jobs doing random search at begining of optimization. Higher values encourages exploration
N_STARTUP_JOBS = int(os.environ.get('N_STARTUP_JOBS', default=10))
# n_EI_candidades: number of config samples draw before select the best.
N_EI_CANDIDATES = int(os.environ.get('N_EI_CANDIDATES', default=24))
NUMBER_ITERATIONS = int(float(
    os.environ.get('NUMBER_ITERATIONS', default='10')))
ITERATIONS_BEFORE_REINFORCE = int(os.environ.get('ITERATIONS_BEFORE_REINFORCE', default='2'))
REINFORCEMENT_RATIO = float(os.environ.get('REINFORCEMENT_RATIO', default='1.0'))
PROBATION_RATIO = float(os.environ.get('PROBATION_RATIO', default='1.0'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', default=time.time()))

WAITING_TIME = int(os.environ.get('WAITING_TIME', default='120'))

# actuator config
NAMESPACE = os.environ.get('NAMESPACE', default='quarkus')
HPA_NAME = os.environ.get('HPA_NAME', default='quarkus-service')

# -- begin -- remove this when delete sampler.py
OBJECTIVE = str(os.environ.get('OBJECTIVE', default='memory_limit'))
SAMPLE_SIZE = float(os.environ.get('SAMPLE_SIZE', default='0.3334'))
# marked for deprecation
AGGREGATION_FUNCTION = os.environ.get('AGGREGATION_FUNCTION', default='sum')  # sum, avg, max, min
QUANTILE = float(os.environ.get('QUANTILE', default='1.0'))
# ------------------------------
# Prom Queries
# ------------------------------
Q_CPU = os.environ.get('Q_CPU', default='f\'sum(rate(container_cpu_usage_seconds_total{{id=~".*kubepods.*",'
                                        'pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",'
                                        'container=""}}[''{self.interval}s]))\'')

Q_CPU_L = os.environ.get('Q_CPU_L', default='f\'sum(sum_over_time(container_spec_cpu_quota{{id=~".*kubepods.*",'
                                            'pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",'
                                            'container=""}}[{self.interval}s])) / avg(sum_over_time('
                                            'container_spec_cpu_period{{id=~".*kubepods.*",pod=~"{self.podname}-.*",'
                                            'name!~".*POD.*",namespace="{self.namespace}",container=""}}[{'
                                            'self.interval}s]))\'')

Q_MEM = os.environ.get('Q_MEM', default='f\'sum(max_over_time(container_memory_working_set_bytes{{id=~".*kubepods.*",'
                                        'pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",'
                                        'container=""}}[{self.interval}s]))\'')

Q_MEM_L = os.environ.get('Q_MEM_L',
                         default='f\'sum(max_over_time(container_spec_memory_limit_bytes{{id=~".*kubepods.*",'
                                 'pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",'
                                 'container=""}}[{self.interval}s]))\'')

Q_THRUPUT = os.environ.get('Q_THRUPUT', default='f\'sum(rate(smarttuning_http_requests_total{{code=~"[2|3]..",'
                                                'pod=~"{self.podname}-.*",namespace="{self.namespace}",name!~".*POD.*"}}'
                                                '[{self.interval}s]))\'')

Q_RESP_TIME = os.environ.get('Q_RESP_TIME', default='f\'sum(rate(smarttuning_http_processtime_seconds_sum{{pod=~"'
                                                    '{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}"}}'
                                                    '[{self.interval}s])) / sum( rate('
                                                    'smarttuning_http_processtime_seconds_count{{pod=~"{self.podname}-.*",'
                                                    'name!~".*POD.*",namespace="{self.namespace}"}}[{self.interval}s]))\'')

Q_ERRORS = os.environ.get('Q_ERRORS', default='f\'sum(rate(smarttuning_http_requests_total{{code=~"[4|5]..",'
                                              'pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}"}}['
                                              '{self.interval}s])) / sum( rate(smarttuning_http_requests_total{{pod=~"{'
                                              'self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}"}}[{'
                                              'self.interval}s]))\'')

Q_REPLICAS = os.environ.get('Q_REPLICAS', default='f\'sum(count(count(sum(rate(container_cpu_usage_seconds_total{{'
                                                  'id=~".*kubepods.*",pod=~"{self.podname}-.*",name!~".*POD.*",container!="",'
                                                  'namespace="{self.namespace}"}}[{self.interval}s])) by (container,pod)) by (pod) > 1) '
                                                  'by (pod))\'')

# ------------------------------
# -- end --

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


__hpaApi = None


def hpaApi() -> kubernetes.client.api.autoscaling_v2beta2_api.AutoscalingV2beta2Api:
    global __hpaApi
    if not __hpaApi:
        __hpaApi = kubernetes.client.api.autoscaling_v2beta2_api.AutoscalingV2beta2Api()
    return __hpaApi


def executor(max_workers: int = None) -> ThreadPoolExecutor:
    global _executor
    if not _executor:
        _executor = ThreadPoolExecutor(max_workers=max_workers)
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


def shutdown():
    mongo().close()
    executor().shutdown(wait=False)
