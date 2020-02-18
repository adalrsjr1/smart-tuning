from pymongo import MongoClient

import requests
import math
import numpy as np


class MongoAccessLayer:
    def __init__(self, url, port, database):
        self._connection = MongoClient(url, port)
        self._database = self._connection[database]

    def connection(self):
        return self._connection

    def close(self):
        self.connection().close()

    def collection(self, name):
        return self._database[name]

    def store(self, data, collection):
        if not isinstance(data, list):
            data = [data]
        return collection.insert_many(data)

    def find(self, query, collection):
        return collection.find(query)


class PrometheusAccessLayer:
    def __init__(self, prometheus_url, prometheus_port):
        self.url = prometheus_url
        self.port = prometheus_port

    def query(self, query, start, end, step, to_round=False):
        response = requests.get(f'http://{self.url}:{self.port}/api/v1/query_range',
                                params={
                                    'query': query,
                                    'start': start,
                                    'end': end,
                                    'step': step
                                })

        if response.json()['status'] == 'success':
            try:
                result = response.json()['data']['result']
                data = []
                if len(result) > 0:
                    values = result[0]['values']
                    if to_round:
                        data = [math.floor(float(value)) for key, value in values]
                    data = [float(value) for key, value in values]
                return PrometheusResponse(np.array(data))
            except Exception as e:
                print(e)
                print(f'query:{response.request.url}, status:{response.json()["status"]}, code:{response.status_code}')
                return PrometheusResponse(np.array([]))
        else:
            print(f'query:{response.request.url}, status:{response.json()["error"]}, code:{response.status_code}')
            return PrometheusResponse(np.array([]))

    def rate(self, expression, start, end, step, to_round=False):
        return self.query(f'rate({expression})[{step}s]', start, end, step)

    def avg(self, expression, start, end, step, to_round=False):
        return self.query(f'avg_over_time({expression}[{step}s])', start, end, step, to_round)

    def std(self, expression, start, end, step, to_round=False):
        return self.query(f'avg_over_time({expression}[{step}s])', start, end, step, to_round)

    def increase(self, expression, start, end, step, to_round=False):
        return self.query(f'increase({expression}[{step}s])', start, end, step, to_round)


class PrometheusResponse:
    def __init__(self, data):
        self.data = data

    def length(self):
        return len(self.data)

    def split(self, n_intervals):
        data_length = self.length()
        # guarantee of n_intervals is always multiple of data_lenght
        # if n_intervals is not multiple, n_intervals will be the closest
        # smaller
        n_intervals = ((data_length - 1) % n_intervals) + 1

        return np.split(self.data, n_intervals)

    def split_and_group(self, n_intervals, lambda_action):
        subintervals = self.split(n_intervals)
        return np.array([lambda_action(interval) for interval in subintervals])

    def group(self, lambda_action):
        return lambda_action(self.data)

    def is_empty(self):
        return len(self.data) == 0
