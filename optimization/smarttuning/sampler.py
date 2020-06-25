from concurrent.futures import Future
import config
import logging
import pandas as pd
import json
from numbers import Number
import math
from prometheus_pandas import query as handler

logger = logging.getLogger(config.SAMPLER_LOGGER)
logger.setLevel(logging.DEBUG)


def __extract_value_from_future__(future, timeout=config.SAMPLING_METRICS_TIMEOUT):
    result = future.result(timeout=timeout)
    metric = result.replace(float('NaN'), 0)
    return metric[0] if not metric.empty else 0

def series_to_dataframe(series: pd.Series):
    rows = []
    key: str
    for key, value in series.items():
        key = key.replace('=', '":', 1).replace('{', '{"').replace(',', ',"')
        key = json.loads(key)
        key.update({'value': value})
        rows.append(key)
    labels = key.keys()

    table = {label: [] for label in labels}
    for row in rows:
        for key, value in row.items():
            table[key].append(value)

    return pd.DataFrame(table)


class Metric:
    def __init__(self,
                 f_cpu: Future = None,
                 cpu: Number = None,
                 f_memory: Future = None,
                 memory: Number = None,
                 f_throughput: Future = None,
                 throughput: Number = None,
                 f_latency: Future = None,
                 latency: Number = None,
                 f_errors: Future = None,
                 errors: Number = None,
                 to_eval=config.OBJECTIVE
                 ):
        self._f_cpu = f_cpu
        self._cpu = cpu
        self._f_memory = f_memory
        self._memory = memory
        self._f_throughput = f_throughput
        self._throughput = throughput
        self._f_latency = f_latency
        self._latency = latency
        self._f_errors = f_errors
        self._errors = errors
        self.to_eval = to_eval

    def cpu(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._cpu is None:
            self._cpu = __extract_value_from_future__(self._f_cpu, timeout)
        return self._cpu

    def memory(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._memory is None:
            self._memory = __extract_value_from_future__(self._f_memory, timeout)
        return self._memory

    def throughput(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._throughput is None:
            self._throughput = __extract_value_from_future__(self._f_throughput, timeout)
        return self._throughput

    def latency(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._latency is None:
            self._latency = __extract_value_from_future__(self._f_latency, timeout)
        return self._latency

    def errors(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._errors is None:
            self._errors = __extract_value_from_future__(self._f_errors, timeout)
        return self._errors

    def __operation__(self, other, op):
        if isinstance(other, Metric):
            logger.debug("op(Metric, Metric)")
            return Metric(cpu=op(self.cpu(), other.cpu()),
                          memory=op(self.memory(), other.memory()),
                          throughput=op(self.throughput(), other.throughput()),
                          latency=op(self.latency(), other.latency()),
                          errors=op(self.errors(), other.errors()))

        if isinstance(other, Number):
            logger.debug("op(Metric, Scalar)")
            return Metric(cpu=op(self.cpu(), other),
                          memory=op(self.memory(), other),
                          throughput=op(self.throughput(), other),
                          latency=op(self.latency(), other),
                          errors=op(self.errors(), other))

        raise TypeError(f'other is {type(other)} and it should be a scalar or a Metric type')

    def serialize(self):
        serialized = self.to_dict()
        serialized.update({'objective': self.objective()})
        return serialized

    def to_dict(self):
        return {
            'cpu': self.cpu(),
            'memory': self.memory(),
            'throughput': self.throughput(),
            'latency': self.latency(),
            'errors': self.errors()
        }

    def __eq__(self, other):
        return self.memory() == other.memory() and \
               self.cpu() == other.cpu() and \
               self.throughput() == other.throughput() and \
               self.latency() == other.latency() and \
               self.errors() == other.errors() and \
               self.objective() == other.objective()

    def __hash__(self):
        return hash((self.memory(), self.cpu(), self.throughput(), self.latency(), self.errors(), self.objective()))

    def __add__(self, other):
        return self.__operation__(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self.__operation__(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self.__operation__(other, lambda a, b: a * b)

    def __floordiv__(self, other):
        return self.__operation__(other, lambda a, b: a.__floordiv__(b))

    def __truediv__(self, other):
        return self.__operation__(other, lambda a, b: a.__truediv__(b))

    def __divmod__(self, other):
        return self.__operation__(other, lambda a, b: a.__divmod__(b))

    def __lt__(self, other):
        return self.objective() < (other.objective() if isinstance(other, Metric) else other)

    def __le__(self, other):
        return self.objective() <= (other.objective() if isinstance(other, Metric) else other)

    def __gt__(self, other):
        return self.objective() > (other.objective() if isinstance(other, Metric) else other)

    def __ge__(self, other):
        return self.objective() >= (other.objective() if isinstance(other, Metric) else other)

    def __repr__(self):
        return f'Metric(cpu={self.cpu()}, memory={self.memory()}, throughput={self.throughput()}, ' \
               f'latency={self.latency()}, errors={self.errors()}, objective={self.objective()})'

    def objective(self) -> float:
        try:
            result = eval(self.to_eval, globals(), self.to_dict()) if self.to_eval else float('inf')
            if math.isnan(result):
                return float('inf')
            return result
        except ZeroDivisionError:
            logger.exception('error metric.objective division by 0')
            return float('inf')


class PrometheusSampler:
    def __init__(self, podname: str, interval: int, executor=config.executor, addr=config.PROMETHEUS_ADDR,
                 port=config.PROMETHEUS_PORT):
        self.client = handler.Prometheus(f'http://{addr}:{port}')
        self.executor = executor
        self.podname = podname
        self.interval = interval

    def __do_sample__(self, query: str) -> Future:
        return self.executor.submit(self.client.query, query)

    def metric(self, quantile=1.0) -> Metric:
        return Metric(
            f_cpu=self.cpu(quantile),
            f_memory=self.memory(quantile),
            f_throughput=self.throughput(quantile),
            f_latency=self.latency(quantile),
            f_errors=self.error(quantile)
        )

    def throughput(self, quantile=1.0) -> Future:
        """ return future<pd.Series>"""
        logger.debug(f'sampling throughput at {self.podname}')
        query = f'quantile({quantile},rate(smarttuning_http_requests_total{{pod=~"{self.podname}",name!~".*POD.*"}}[{self.interval}s]))'
        return self.__do_sample__(query)

    def error(self, quantile=1.0) -> Future:
        logger.debug(f'sampling errors rate at {self.podname}')
        query = f'quantile({quantile}, rate(smarttuning_http_requests_total{{code=~"5..",pod=~"{self.podname}",name!~".*POD.*"}}[{self.interval}s])) /' \
                f'quantile({quantile}, rate(smarttuning_http_requests_total{{pod=~"{self.podname}",name!~".*POD.*"}}[{self.interval}s]))'
        return self.__do_sample__(query)

    def latency(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the latency_sum/latency_count rate of an specific pod"""
        logger.debug(f'sampling latency at {self.podname}')
        query = f'quantile({quantile}, rate(smarttuning_http_latency_seconds_sum{{pod=~"{self.podname}",name!~".*POD.*"}}[{self.interval}s])) / ' \
                f'quantile({quantile}, rate(smarttuning_http_latency_seconds_count{{pod=~"{self.podname}",name!~".*POD.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def memory(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the memory (bytes) quantile over time of an specific pod
            :param quantile a value 0.0 - 1.0
        """
        logger.debug(f'sampling memory at {self.podname}')
        query = f'quantile({quantile}, quantile_over_time({quantile},container_memory_working_set_bytes{{pod=~"{self.podname}",name!~".*POD.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def cpu(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the CPU (milicores) rate over time of an specific pod
        """
        logger.debug(f'sampling cpu at {self.podname}')
        query = f'quantile({quantile},rate(container_cpu_usage_seconds_total{{pod=~"{self.podname}",name!~".*POD.*"}}[{self.interval}s]))'
        return self.__do_sample__(query)

    def workload(self) -> Future:
        """
        :return: Workload object. Note that urls are grouped in case of using <Path Parameters>, e.g.,

        the urls:

        /my/url/using/path-parameter/uid153@email.com
        /my/url/using/path-parameter/uid188@email.com
        /my/url/using/path-parameter/uid174@email.com

        are grouped into /my/url/using/path-parameter/uid153@email.com

        """
        logger.debug(f'sampling urls at {self.podname}')

        query = f'sum by (path)(rate(smarttuning_http_requests_total{{pod=~"{self.podname}"}}[{self.interval}s]))' \
                f' / ignoring ' \
                f'(path) group_left sum(rate(smarttuning_http_requests_total{{pod=~"{self.podname}"}}[{self.interval}s]))'

        return self.__do_sample__(query)