from collections import namedtuple

import requests
import numpy as np

class PrometheusAccessLayer:
    def __init__(self, prometheus_url:str, prometheus_port:str):
        self.url = prometheus_url
        self.port = prometheus_port

    def query(self, query, timestamp=None):
        url = f'http://{self.url}:{self.port}/api/v1/query'

        params = {'query': query}

        if timestamp:
            params.update({'time': timestamp})

        response = requests.get(url, params=params)

        return PrometheusResponse(response)

    def query_range(self, query, start, end, step):
        response = requests.get(f'http://{self.url}:{self.port}/api/v1/query_range',
                                params={
                                    'query': query,
                                    'start': start,
                                    'end': end,
                                    'step': step
                                })

        return PrometheusResponse(response)


class PrometheusResponse:

    # {
    #   "status": "success" | "error",
    #   "data": <data>,
    #
    #   // Only set if status is "error". The data field may still hold
    #   // additional data.
    #   "errorType": "<string>",
    #   "error": "<string>",
    #
    #   // Only if there were warnings while executing the request.
    #   // There will still be data in the data field.
    #   "warnings": ["<string>"]
    # }

    SUCCESS = 'success'
    ERROR = 'error'
    MATRIX = 'matrix'
    VECTOR = 'vector'
    SCALAR = 'scalar'
    STRING = 'string'

    def __init__(self, response):
        self.__url = response.request.url
        self.__status_code = response.status_code

        self.json = response.json()

    def __str__(self):
        if PrometheusResponse.SUCCESS == self.status():
            return self.np_values().__str__()
        else:
            return self.error()

    def status(self):
        return self.json['status']

    def error(self):
        return f'\{"url":"{self.__url}", "status":"{self.status()}", "code":"{self.__status_code}"\}'

    def data(self):
        Data = namedtuple('Data', ['result_type', 'result'])

        result_type = ''
        result = []

        if self.status() == self.SUCCESS and self.json.get('data'):
            result_type = self.json['data']['resultType']
            result = self.json['data']['result']

        return Data(result_type=result_type, result=result)

    def result_type(self):
        return self.data().result_type

    def result(self, index=None):
        if index:
            return self.data().result[index]
        return self.data().result

    def metric(self, index=None):
        if index:
            return self.result(index)['metric']
        return [result['metric'] for result in self.result()]

    def value(self, index=None):
        Value = namedtuple('Value', ['timestamp', 'value'])
        if index:
            return Value(timestamp=self.result(index)['value'][0], value=self.result(index)['value'][1])
        return [Value(timestamp=result['value'][0], value=result['value'][1]) for result in self.result()]

    def np_values(self):
        return np.array([result['value'][1] for result in self.result()])

    def error_type(self):
        _error_type = self.json['errorType']
        if _error_type:
            return _error_type
        return ''

    def error(self):
        _error = self.json['error']
        if _error:
            return _error
        return ''

    def warnings(self):
        return self.json['warnings']