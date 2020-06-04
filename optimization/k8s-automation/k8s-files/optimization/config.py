import os
import time
from concurrent.futures import ThreadPoolExecutor, wait as ThreadWait, ALL_COMPLETED as FUTURE_ALL_COMPLETED

from pymongo import MongoClient

def print_config():
    for item in globals().items():
        if item[0].isupper():
            print(item)

print('\n *** loading config ***\n')
# test config
MOCK = eval(os.environ.get('MOCK', default='True'))

# mongo config
MONGO_ADDR = os.environ.get('MONGO_ADDR', default='127.0.0.1')
MONGO_PORT = int(os.environ.get('MONGO_PORT', default='30027'))
MONGO_DB = os.environ.get('MONGO_DB', default='smarttuning')

# prometheus config
PROMETHEUS_ADDR = os.environ.get('PROMETHEUS_ADDR', default='localhost')
PROMETHEUS_PORT = os.environ.get('PROMETHEUS_PORT', default='30090')

# classification config
K = int(os.environ.get('K', default='3'))
DISTANCE_METHOD = os.environ.get('DISTANCE_METHOD', default='hellinger')

# optimization config
SEARCHSPACE_PATH = os.environ.get('SEARCHSPACE_PATH',default='')
BAYESIAN = eval(os.environ.get('OPTIMIZATION_METHOD', default='True'))
NUMBER_ITERATIONS = int(os.environ.get('NUMBER_ITERATIONS', default='3'))
METRIC_THRESHOLD = float(os.environ.get('METRIC_THRESHOLD', default='0.2'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', default=time.time()))

# sampling config
SAMPLE_SIZE = float(os.environ.get('SAMPLE_SIZE', default='1.0'))
WAITING_TIME = int(os.environ.get('WAITING_TIME', default='2'))
POD_REGEX = os.environ.get('POD_REGEX', '.*tuning.*')
POD_PROD_REGEX = os.environ.get('POD_PROD_REGEX', '.*tuningprod.*')

# actuator config
CONFIGMAP_NAME = os.environ.get('CONFIGMAP_NAME', default='tuning-config')
CONFIGMAP_PROD_NAME = os.environ.get('CONFIGMAP_PROD_NAME', default='tuning-config')
NAMESPACE = os.environ.get('NAMESPACE', 'default')
NAMESPACE_PROD = os.environ.get('NAMESPACE_PROD', 'default')

# deprecated -- to remove
# CONFIG_PATH = os.environ.get('CONFIG_PATH', default='/etc')
# REGISTER_SERVER_PORT = int(os.environ.get('REGISTER_SERVER_PORT', default='5000'))
# REGISTER_SERVER_ADDR = os.environ.get('REGISTER_SERVER_ADDR', default='0.0.0.0')
# SYNC_PORT = int(os.environ.get('SYNC_PORT', default='5000'))


print_config()
print('\n *** config loaded *** \n')

executor = ThreadPoolExecutor()
client = MongoClient(MONGO_ADDR, MONGO_PORT)

def shutdown():
    client.close()
    executor.shutdown()


