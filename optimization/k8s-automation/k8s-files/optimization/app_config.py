import os

MOCK = bool(os.environ.get('MOCK', False))
MONGO_ADDR = os.environ.get('MONGO_ADDR', 'localhost')
MONGO_PORT = int(os.environ.get('MONGO_PORT', '30027'))
MONGO_DB = os.environ.get('MONGO_DB', 'smarttuning')
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