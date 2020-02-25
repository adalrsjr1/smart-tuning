import os
import json
import sys

from os import environ

from pathlib import Path
from dotenv import load_dotenv

from common.dataaccess import PrometheusAccessLayer
from common.dataaccess import MongoAccessLayer
from common import timeutil

if len(sys.argv) > 1:
    env_path = Path(sys.argv[1])
    load_dotenv(verbose=True, dotenv_path=env_path)

HIST_QUERY = environ.get('HIST_QUERY').split(',')
URL_LABELS = environ.get('URL_LABELS').split(',')
MEAN_QUERY = environ.get('MEAN_QUERY').split(',')
STD_QUERY = environ.get('STD_QUERY').split(',')
METRIC = environ.get('METRIC').split(',')

PROMETHEUS_ENDPOINT = environ.get('PROMETHEUS_ENDPOINT')
PROMETHEUS_PORT = int(environ.get('PROMETHEUS_PORT'))
QUERY_STEP = int(environ.get('QUERY_STEP'))
MONGO_ENDPOINT = environ.get('MONGO_ENDPOINT')
MONGO_PORT = int(environ.get('MONGO_PORT'))
MONGO_DB = environ.get('MONGO_DB')
MONGO_COLLECTION = environ.get('MONGO_COLLECTION')
MONGO = environ.get('MONGO', '')

KEEP_DATA = bool(environ.get('KEEP_DATA', False))
BUFFER = int(environ.get('BUFFER', 0))
SCRAP_INTERVAL = int(environ.get('SCRAP_INTERVAL', 5))

METADATA = {
    'application': environ.get('APPLICATION')
}


def main():
    global MEAN_QUERY
    global STD_QUERY
    global QUERY_STEP
    global PROMETHEUS_ENDPOINT
    global PROMETHEUS_PORT
    global MONGO_ENDPOINT
    global MONGO_PORT
    global MONGO_DB
    global MONGO_COLLECTION
    global MONGO
    global KEEP_DATA
    global BUFFER
    global SCRAP_INTERVAL

    prometheus = PrometheusAccessLayer(PROMETHEUS_ENDPOINT, PROMETHEUS_PORT)

    mongo_collection = ''
    if KEEP_DATA:
        if BUFFER > 0 and MONGO_ENDPOINT and MONGO_PORT and MONGO_DB and MONGO_COLLECTION:
            MONGO = MongoAccessLayer(MONGO_ENDPOINT, MONGO_PORT, MONGO_DB)
            mongo_collection = MONGO.collection(MONGO_COLLECTION)

    buffer = []
    while True:
        end = timeutil.now()

        data = []
        for metric, mean, std in zip(METRIC, MEAN_QUERY, STD_QUERY):

            query_mean = prometheus.query(mean).np_values()
            query_std = prometheus.query(std).np_values()
            if query_mean.size > 0 and query_std.size > 0:
                data.append({'metric': metric, 'mean': query_mean[0], 'std': query_std[0]})
            else:
                print('{not saving metrics mean and or std is empty}')

        histogram = {}
        for url_label, hist_query in zip(URL_LABELS, HIST_QUERY):
            values = prometheus.query(hist_query).np_values()
            if values.size > 0:
                bucket = prometheus.query(hist_query).np_values()[0]
                histogram.update({url_label: bucket})

        data = {
            'metrics': {'n_samples': QUERY_STEP // SCRAP_INTERVAL, 'values': data},
            'histogram': histogram,
            'start': end - QUERY_STEP,
            'end': end,
            'step': QUERY_STEP
        }

        data.update(METADATA)

        buffer.append(data)
        print(json.JSONEncoder().encode(data))

        if BUFFER != 0:
            if MONGO and len(buffer) >= BUFFER:
                string = 'saving {0} datapoints'.format(len(buffer))
                print('{' + string + '}')
                MONGO.store(buffer, mongo_collection)
                buffer = []
        else:
            buffer = []

        timeutil.sleep(QUERY_STEP)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
        finally:
            if isinstance(MONGO, MongoAccessLayer) and MONGO:
                MONGO.close()
    finally:
        if isinstance(MONGO, MongoAccessLayer) and MONGO:
            MONGO.close()

