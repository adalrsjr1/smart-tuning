import os
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, wait as ThreadWait, ALL_COMPLETED as FUTURE_ALL_COMPLETED
executor = ThreadPoolExecutor(4)

MOCK = bool(os.environ.get('MOCK', True))
MONGO_ADDR = os.environ.get('MONGO_ADDR', '127.0.0.1')
MONGO_PORT = int(os.environ.get('MONGO_PORT', '30027'))
MONGO_DB = os.environ.get('MONGO_DB', 'smarttuning')
client = MongoClient(MONGO_ADDR, MONGO_PORT)


K = int(os.environ.get('K', '3'))
CONFIG_PATH = os.environ.get('CONFIG_PATH', '/etc')
SEARCHSPACE_PATH = os.environ.get('SEARCHSPACE_PATH','')
WAITING_TIME = int(os.environ.get('WAITING_TIME', '2'))
CONFIGMAP_NAME = os.environ.get('CONFIGMAP_NAME', 'tuning-config')
NAMESPACE = os.environ.get('NAMESPACE', 'default')
DISTANCE_METHOD = os.environ.get('DISTANCE_METHOD', 'hellinger')
PROMETHEUS_ADDR = os.environ.get('PROMETHEUS_ADDR', 'localhost')
PROMETHEUS_PORT = os.environ.get('PROMETHEUS_PORT', '30090')
DISTANCE_METHOD = 'hellinger'
NUMBER_ITERATIONS = int(os.environ.get('NUMBER_ITERATIONS', '3'))
METRIC_THRESHOLD = float(os.environ.get('METRIC_THRESHOLD', '0.2'))
REGISTER_SERVER_PORT = int(os.environ.get('REGISTER_SERVER_PORT', '5000'))
REGISTER_SERVER_ADDR = os.environ.get('REGISTER_SERVER_ADDR', '0.0.0.0')
SYNC_PORT = int(os.environ.get('SYNC_PORT', '5000'))
