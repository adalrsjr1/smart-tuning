from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from concurrent.futures import Future
from numbers import Number

import networkx as nx
import pandas as pd
from prometheus_pandas import query as handler

import config

logger = logging.getLogger(config.SAMPLER_LOGGER)
logger.setLevel(logging.DEBUG)


def __extract_value_from_future__(future, timeout=config.SAMPLING_METRICS_TIMEOUT):
    try:
        result = future.result(timeout=timeout)
        metric = result.replace(float('NaN'), 0)
        return metric[0] if not metric.empty else 0
    except Exception as e:
        logger.error(e)
        return float('nan')


def __extract_in_out_balance__(podname, future, timeout=config.SAMPLING_METRICS_TIMEOUT):
    return float('nan')
    try:
        df = series_to_dataframe(future.result(timeout=timeout))
        table = {}
        links_df = df.copy()
        for i, item in df.iterrows():
            splitted_ip = item['instance'].split(':')[0]
            df.loc[i, ('instance')] = splitted_ip
            k = item['pod'].count('-')

            # k - 2 to remove the k8s's auto-generated uuid
            table[splitted_ip] = '-'.join(item['pod'].split('-', maxsplit=k - 1)[:k - 1])

        # replace IPs with service names
        for i, instance in enumerate(links_df['instance']):
            links_df.loc[i, ('dst')] = table.get(links_df.loc[i, ('dst')], links_df.loc[i, ('dst')])
            links_df.loc[i, ('src')] = table.get(links_df.loc[i, ('src')], links_df.loc[i, ('src')])

        # # transform table into graph
        G = nx.DiGraph()
        ip_regex = re.compile("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
        for i, row in links_df.iterrows():
            if not ip_regex.match(row['src']) and not ip_regex.match(row['dst']):
                v = float(row['value'])
                if math.isnan(v):
                    continue
                try:
                    path = row["path"]
                except KeyError:
                    path = ''
                if G.has_edge(row['src'], row['dst']):
                    weight = G[row['src']][row['dst']]['weight']
                    G[row['src']][row['dst']]['weight'] += v
                    G[row['src']][row['dst']]['label'] = f"[{G[row['src']][row['dst']]['weight']:.2f}]"
                else:
                    G.add_edge(row['src'], row['dst'], weight=v, label=f'[{v:.2f}]')

        selected = [node for node in G.nodes if re.match(podname, node)][0]

        result = {'in': 0, 'out': 0}
        for edge in G.edges.items():
            if selected in edge[0]:
                if selected == edge[0][0]:
                    result['out'] += edge[1]['weight']
                else:
                    result['in'] += edge[1]['weight']
        return result['in']
    except:
        logger.exception('cannot calculate in/out balance')
        return float('nan')


def series_to(series: pd.Series, container):
    rows = []
    key: str
    for key, value in series.items():
        key = parser_to_dict(key)
        key.update({'value': value})
        rows.append(key)
    # labels = series.keys()

    table = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            table[key].append(value)

    if container is dict:
        return table
    return container(table)


def parser_to_dict(string: str) -> dict:
    string = re.sub('[{}"]', '', string)
    l_string = string.split(',')
    new_dict = {}
    for token in l_string:
        splited = token.split('=', 1)
        new_dict[splited[0]] = splited[1]
    return new_dict


def series_to_dataframe(series: pd.Series):
    return series_to(series, pd.DataFrame)


def series_to_dict(series: pd.Series):
    return series_to(series, dict)


class Metric:

    def __init__(self,
                 name='',
                 f_cpu: Future = None,
                 cpu: Number = None,
                 f_memory: Future = None,
                 memory: Number = None,
                 f_memory_limit: Future = None,
                 memory_limit: Number = None,
                 f_throughput: Future = None,
                 throughput: Number = None,
                 f_process_time: Future = None,
                 process_time: Number = None,
                 f_errors: Future = None,
                 errors: Number = None,
                 f_in_out=None,
                 to_eval='',
                 in_out=None,
                 restarts=0):

        self.name = name
        self._f_cpu = f_cpu
        self._cpu = cpu
        self._f_memory = f_memory
        self._memory = memory
        self._f_memory_limit = f_memory_limit
        self._memory_limit = memory_limit
        self._f_throughput = f_throughput
        self._throughput = throughput
        self._f_process_time = f_process_time
        self._process_time = process_time
        self._f_errors = f_errors
        self._errors = errors
        self._f_in_out = f_in_out
        self._in_out = in_out
        self.to_eval = to_eval if len(to_eval) > 0 else config.OBJECTIVE
        self._restarts = restarts

    _instance = None

    @staticmethod
    def zero():
        if not Metric._instance:
            Metric._instance = Metric(name='', cpu=0, memory=0, memory_limit=0, throughput=0, process_time=0, errors=0,
                                      to_eval='0', restarts=0)
        return Metric._instance

    def cpu(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._cpu is None:
            self._cpu = __extract_value_from_future__(self._f_cpu, timeout)
        return self._cpu

    def memory(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._memory is None:
            self._memory = __extract_value_from_future__(self._f_memory, timeout)
        return self._memory

    def memory_limit(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._memory_limit is None:
            self._memory_limit = __extract_value_from_future__(self._f_memory_limit, timeout)
        return self._memory_limit

    def throughput(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._throughput is None:
            self._throughput = __extract_value_from_future__(self._f_throughput, timeout)
        return self._throughput

    def process_time(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._process_time is None:
            self._process_time = __extract_value_from_future__(self._f_process_time, timeout)
        return self._process_time

    def errors(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._errors is None:
            self._errors = __extract_value_from_future__(self._f_errors, timeout)
        return self._errors

    def in_out(self, timeout=config.SAMPLING_METRICS_TIMEOUT):
        if self._in_out is None:
            self._in_out = __extract_in_out_balance__(self.name, self._f_in_out, timeout)
        return self._in_out

    def restarts(self):
        return self._restarts

    def set_restarts(self, n):
        self._restarts = n

    def __operation__(self, other, op):
        if isinstance(other, Metric):
            logger.debug("op(Metric, Metric)")
            return Metric(
                name=f'{self.name}_{other.name}',
                cpu=op(self.cpu(), other.cpu()),
                memory=op(self.memory(), other.memory()),
                memory_limit=op(self.memory_limit(), other.memory_limit()),
                throughput=op(self.throughput(), other.throughput()),
                process_time=op(self.process_time(), other.process_time()),
                errors=op(self.errors(), other.errors()),
                in_out=op(self.in_out(), other.in_out()),
                to_eval=self.to_eval,
                restarts=op(self.restarts(), other.restarts())
            )

        if isinstance(other, Number):
            logger.debug("op(Metric, Scalar)")
            return Metric(
                name=f'{self.name}_{other}',
                cpu=op(self.cpu(), other),
                memory=op(self.memory(), other),
                memory_limit=op(self.memory_limit(), other),
                throughput=op(self.throughput(), other),
                process_time=op(self.process_time(), other),
                errors=op(self.errors(), other),
                in_out=op(self.in_out(), other),
                to_eval=self.to_eval,
                restarts=op(self.restarts(), other),
            )

        raise TypeError(f'other is {type(other)} and it should be a scalar or a Metric type')

    def serialize(self):
        serialized = self.to_dict()
        serialized.update({'objective': self.objective()})
        return serialized

    def to_dict(self):
        return {
            'name': self.name,
            'cpu': self.cpu(),
            'memory': self.memory(),
            'memory_limit': self.memory_limit(),
            'throughput': self.throughput(),
            'process_time': self.process_time(),
            'in_out': self.in_out(),
            'errors': self.errors(),
            'restarts': self.restarts()
        }

    def __eq__(self, other: Metric):
        """
        retursn False if some value is NaN. According to IEEE 754 nan cannot be compared
        see: https://bugs.python.org/issue28579
        """
        return self.__eval_values(self.memory(), other.memory()) and \
               self.__eval_values(self.memory_limit(), other.memory_limit()) and \
               self.__eval_values(self.cpu(), other.cpu()) and \
               self.__eval_values(self.throughput(), other.throughput()) and \
               self.__eval_values(self.process_time(), other.process_time()) and \
               self.__eval_values(self.errors(), other.errors()) and \
               self.__eval_values(self.in_out(), other.in_out()) and \
               self.__eval_values(self.objective(), other.objective()) and \
               self.__eval_values(self.restarts(), other.restarts())


    def __eval_values(self, a:float, b:float):
        if math.isfinite(a) and math.isfinite(b):
            return a == b

        if math.isnan(a) and math.isnan(b):
            return True

        if math.isinf(a) and math.isinf(b):
            return True

    def __hash__(self):
        return hash((self.memory(), self.cpu(), self.throughput(), self.process_time(), self.in_out(), self.errors(),
                     self.objective(), self.restarts()))

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
        return f'{{"cpu":{self.cpu()}, "memory":{self.memory()}, "memory_limit":{self.memory_limit()}, "throughput":{self.throughput()}, ' \
               f'"process_time":{self.process_time()}, "in_out":{self.in_out()}, "errors":{self.errors()}, "restarts":{self.restarts()}, "objective":{self.objective()}}}'

    def objective(self) -> float:
        try:
            result = eval(self.to_eval, globals(), self.to_dict()) if self.to_eval else float('inf')
            if math.isnan(result):
                logger.warning(f'objective at {self.name} is NaN -> INF')
                return float('inf')
            return result
        except:
            logger.exception('cannot correctly evaluate the objective expression')
            return float('inf')


class PrometheusSampler:
    def __init__(self, podname: str, interval: int, namespace: str = config.NAMESPACE, executor=config.executor(),
                 addr=config.PROMETHEUS_ADDR,
                 port=config.PROMETHEUS_PORT, api_url=''):
        if not api_url:
            api_url = f'http://{addr}:{port}'

        self.client = handler.Prometheus(api_url)
        self.executor = executor
        self.podname = podname
        self.interval = int(interval)
        self.namespace = namespace

    @property
    def interval(self) -> int:
        return self._interval

    @interval.setter
    def interval(self, value: int):
        self._interval = int(value)

    def __do_sample__(self, query: str) -> Future:
        logger.debug(f'sampling:{query}')
        return self.executor.submit(self.client.query, query)

    def metric(self, quantile=1.0) -> Metric:
        return Metric(
            name=self.podname,
            f_cpu=self.cpu(quantile),
            f_memory=self.memory(quantile),
            f_memory_limit=self.memory_limit(quantile),
            f_throughput=self.throughput(quantile),
            f_process_time=self.process_time(quantile),
            f_in_out=self.in_out(quantile),
            f_errors=self.error(quantile))

    def throughput(self, quantile=1.0) -> Future:
        """ return future<pd.Series>"""
        logger.debug(f'sampling throughput at {self.podname}-.* in namespace {self.namespace}')
        query = f'sum(rate(smarttuning_http_requests_total{{code=~"[2|3]..",namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'
        return self.__do_sample__(query)

    def error(self, quantile=1.0) -> Future:
        logger.debug(f'sampling errors rate at {self.podname}-.* in {self.namespace}')
        query = f'sum(rate(smarttuning_http_requests_total{{code=~"[4|5]..",namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s])) /' \
                f'sum(rate(smarttuning_http_requests_total{{namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'
        return self.__do_sample__(query)

    def process_time(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the processtime_sum/processtime_count rate of an specific pod"""
        logger.debug(f'sampling process time at {self.podname}.* in {self.namespace}')
        query = f'sum(rate(smarttuning_http_processtime_seconds_sum{{namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s])) / ' \
                f'sum(rate(smarttuning_http_processtime_seconds_count{{namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def memory(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the memory (bytes) quantile over time of an specific pod
            :param quantile a value 0.0 - 1.0
        """
        # The better metric is container_memory_working_set_bytes as this is what the OOM killer is watching for.
        logger.debug(f'sampling memory at {self.podname}.* in {self.namespace}')
        # container_spec_memory_limit_bytes
        query = f'sum(max_over_time(container_memory_working_set_bytes{{id=~".kubepods.*",namespace="{self.namespace}", container!="",pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def memory_limit(self, quantily=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the memory limit (bytes) quantile over time of an specific pod
            :param quantile a value 0.0 - 1.0
        """
        query = f'sum(max_over_time(container_spec_memory_limit_bytes{{id=~".kubepods.*",namespace="{self.namespace}", container!="",pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def cpu(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the CPU (milicores) rate over time of an specific pod
        """
        logger.debug(f'sampling cpu at {self.podname}-.* in {self.namespace}')
        query = f'sum(rate(container_cpu_usage_seconds_total{{id=~".kubepods.*",namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'

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
        logger.debug(f'sampling urls at {self.podname}-.* in {self.namespace}')

        # values between 0.0 and 1.0
        query = f'sum by (path)(rate(smarttuning_http_requests_total{{namespace="{self.namespace}", pod=~"{self.podname}-.*"}}[{self.interval}s]))' \
                f' / ignoring ' \
                f'(path) group_left sum(rate(smarttuning_http_requests_total{{namespace="{self.namespace}", pod=~"{self.podname}-.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def in_out(self, quantile=1.0) -> Future:
        logger.debug(f'sampling in_out R at {self.podname}-.* in {self.namespace}')
        is_training = config.PROXY_TAG in self.podname
        if is_training:
            query = f'sum(rate(in_http_requests_total{{namespace="{self.namespace}", pod=~".*{config.PROXY_TAG}.*",name!~".*POD.*"}}[{self.interval}s])) by (pod, src, dst, instance, service) /' \
                    f'sum(rate(out_http_requests_total{{namespace="{self.namespace}", pod=~".*{config.PROXY_TAG}.*",name!~".*POD.*"}}[{self.interval}s])) by (pod, src,  dst, instance, service) '
        else:
            query = f'sum(rate(in_http_requests_total{{namespace="{self.namespace}", pod!~".*{config.PROXY_TAG}.*",name!~".*POD.*"}}[{self.interval}s])) by (pod, src, dst, instance, service) /' \
                    f'sum(rate(out_http_requests_total{{namespace="{self.namespace}", pod!~".*{config.PROXY_TAG}.*",name!~".*POD.*"}}[{self.interval}s])) by (pod, src,  dst, instance, service) '

        return self.__do_sample__(query)


if __name__ == '__main__':
    s = PrometheusSampler('acmeair-nginxservicesmarttuning', config.WAITING_TIME * config.SAMPLE_SIZE)
    # s = PrometheusSampler("acmeair-flightservicesmarttuning.*", config.WAITING_TIME * config.SAMPLE_SIZE)
    timeout = 10
    # print(s.cpu().result(timeout=timeout))
    # print(s.memory().result(timeout=timeout))
    # print(s.throughput().result(timeout=timeout))
    # print(s.process_time().result(timeout=timeout))
    # print(s.error().result(timeout=timeout))
    # print(s.workload().result(timeout=timeout))
    # print(s.in_out().result(timeout=timeout))
    print(s.metric().objective())
