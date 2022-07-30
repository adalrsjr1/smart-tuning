from __future__ import annotations
from concurrent.futures import Future
import logging
import pandas as pd
import re
from numbers import Number
from collections import defaultdict
import math
import time
import traceback
from prometheus_pandas import query as handler


def __extract_value_from_future__(future, timeout):
    try:
        result = future.result(timeout=timeout)
        if isinstance(result, pd.DataFrame):
            return result.fillna(0)
        metric = result.replace(float('NaN'), 0)
        return metric[0] if not metric.empty else 0
    except Exception as e:
        traceback.print_exc()
        return float('nan')

def __extract_in_out_balance__(podname, future, timeout):
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
            table[splitted_ip] = '-'.join(item['pod'].split('-', maxsplit=k-1)[:k-1])

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

def parser_to_dict(string:str) -> dict:
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

    def __init__(self, name='', f_cpu: Future = None, cpu: Number = None, f_memory: Future = None,
                 memory: Number = None, f_throughput: Future = None, throughput: Number = None,
                 f_process_time: Future = None, process_time: Number = None, f_errors: Future = None,
                 errors: Number = None, f_in_out=None, to_eval=None, in_out=None, timeout=10):
        self.name = name
        self._f_cpu = f_cpu
        self._cpu = cpu
        self._f_memory = f_memory
        self._memory = memory
        self._f_throughput = f_throughput
        self._throughput = throughput
        self._f_process_time = f_process_time
        self._process_time = process_time
        self._f_errors = f_errors
        self._errors = errors
        self._f_in_out = f_in_out
        self._in_out = in_out
        self.to_eval = to_eval
        self.timeout = timeout

    _instance = None

    @staticmethod
    def zero():
        if not Metric._instance:
            Metric._instance = Metric(name='', cpu=0, memory=0, throughput=0, process_time=0, errors=0, to_eval='0')
        return Metric._instance

    def cpu(self):
        if self._cpu is None:
            self._cpu = __extract_value_from_future__(self._f_cpu, self.timeout)
        return self._cpu

    def memory(self):
        if self._memory is None:
            self._memory = __extract_value_from_future__(self._f_memory, self.timeout)
        return self._memory

    def throughput(self):
        if self._throughput is None:
            self._throughput = __extract_value_from_future__(self._f_throughput, self.timeout)
        return self._throughput

    def process_time(self):
        if self._process_time is None:
            self._process_time = __extract_value_from_future__(self._f_process_time, self.timeout)
        return self._process_time

    def errors(self):
        if self._errors is None:
            self._errors = __extract_value_from_future__(self._f_errors, self.timeout)
        return self._errors

    def in_out(self):
        if self._in_out is None:
            self._in_out = __extract_in_out_balance__(self.name, self._f_in_out, self.timeout)
        return self._in_out

    def __operation__(self, other, op):
        if isinstance(other, Metric):
            return Metric(name=f'{self.name}_{other.name}', cpu=op(self.cpu(), other.cpu()), memory=op(self.memory(), other.memory()),
                          throughput=op(self.throughput(), other.throughput()),
                          process_time=op(self.process_time(), other.process_time()),
                          errors=op(self.errors(), other.errors()), in_out=op(self.in_out(), other.in_out()))

        if isinstance(other, Number):
            return Metric(name=f'{self.name}_{other}', cpu=op(self.cpu(), other), memory=op(self.memory(), other),
                          throughput=op(self.throughput(), other), process_time=op(self.process_time(), other),
                          errors=op(self.errors(), other), in_out=op(self.in_out(), other))

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
            'throughput': self.throughput(),
            'process_time': self.process_time(),
            'in_out': self.in_out(),
            'errors': self.errors()
        }

    def __eq__(self, other:Metric):
        """
        retursn False if some value is NaN. According to IEEE 754 nan cannot be compared
        see: https://bugs.python.org/issue28579
        """
        return self.memory() == other.memory() and \
               self.cpu() == other.cpu() and \
               self.throughput() == other.throughput() and \
               self.process_time() == other.process_time() and \
               self.errors() == other.errors() and \
               self.in_out() == other.in_out() and \
               self.objective() == other.objective()

    def __hash__(self):
        return hash((self.memory(), self.cpu(), self.throughput(), self.process_time(), self.in_out(), self.errors(), self.objective()))

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
        return f'{{"cpu":{self.cpu()}, "memory":{self.memory()}, "throughput":{self.throughput()}, ' \
               f'"process_time":{self.process_time()}, "in_out":{self.in_out()}, "errors":{self.errors()}, "objective":{self.objective()}}}'

    def objective(self) -> float:
        try:
            result = eval(self.to_eval, globals(), self.to_dict()) if self.to_eval else float('inf')
            if isinstance(result, pd.DataFrame):
                return result.fillna(float('inf'))
            if math.isnan(result):
                return float('inf')
            return result
        except ZeroDivisionError:
            return float('inf')


class PrometheusSampler:
    def __init__(self, podname: str, interval: int, objective:str, timeout:int, namespace:str, executor, addr,
                 port, api_url='', query_range:bool=False, start:int=0, end:int=0,):
        if not api_url:
            api_url = f'http://{addr}:{port}'

        self.client = handler.Prometheus(api_url)
        self.executor = executor
        self.podname = podname
        self.interval = int(interval)
        self.namespace = namespace
        self.objective = objective
        self.timeout = timeout

        self.query_range = query_range
        self.start = start
        self.end = end,

    def __do_sample__(self, query: str) -> Future:
        if self.query_range:
            return self.executor.submit(self.client.query_range, query, self.start, self.end, str(self.interval)+'s')
        return self.executor.submit(self.client.query, query)

    def metric(self, to_eval='', quantile=1.0) -> Metric:
        return Metric(name=self.podname, timeout=self.timeout, f_cpu=self.cpu(quantile), f_memory=self.memory(quantile), f_throughput=self.throughput(quantile),
                      f_process_time=self.process_time(quantile), f_in_out=self.in_out(quantile), f_errors=self.error(quantile), to_eval=self.objective)

    def throughput(self, quantile=1.0) -> Future:
        """ return future<pd.Series>"""
        query = f'sum(rate(smarttuning_http_requests_total{{code=~"[2|3]..",namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'
        return self.__do_sample__(query)

    def error(self, quantile=1.0) -> Future:
        query = f'sum(rate(smarttuning_http_requests_total{{code=~"[4|5]..",namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s])) /' \
                f'sum(rate(smarttuning_http_requests_total{{namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'
        return self.__do_sample__(query)

    def process_time(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the processtime_sum/processtime_count rate of an specific pod"""
        query = f'sum(rate(smarttuning_http_processtime_seconds_sum{{namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s])) / ' \
                f'sum(rate(smarttuning_http_processtime_seconds_count{{namespace="{self.namespace}", pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def memory(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the memory (bytes) quantile over time of an specific pod
            :param quantile a value 0.0 - 1.0
        """
        # The better metric is container_memory_working_set_bytes as this is what the OOM killer is watching for.
        query = f'sum(max_over_time(container_memory_working_set_bytes{{id=~".kubepods.*",namespace="{self.namespace}", container!="",pod=~"{self.podname}-.*",name!~".*POD.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def cpu(self, quantile=1.0) -> Future:
        """ return a concurrent.futures.Future<pandas.Series> with the CPU (milicores) rate over time of an specific pod
        """
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

        query = f'sum by (path)(rate(smarttuning_http_requests_total{{namespace="{self.namespace}", pod=~"{self.podname}-.*"}}[{self.interval}s]))' \
                f' / ignoring ' \
                f'(path) group_left sum(rate(smarttuning_http_requests_total{{namespace="{self.namespace}", pod=~"{self.podname}-.*"}}[{self.interval}s]))'

        return self.__do_sample__(query)

    def in_out(self, quantile=1.0) -> Future:
        # is_training = config.PROXY_TAG in self.podname
        # if is_training:
        if False:
            query = f'sum(rate(in_http_requests_total{{namespace="{self.namespace}", pod=~".*{config.PROXY_TAG}.*",name!~".*POD.*"}}[{self.interval}s])) by (pod, src, dst, instance, service) /' \
                    f'sum(rate(out_http_requests_total{{namespace="{self.namespace}", pod=~".*{config.PROXY_TAG}.*",name!~".*POD.*"}}[{self.interval}s])) by (pod, src,  dst, instance, service) '
        else:
            query = f'sum(rate(in_http_requests_total{{namespace="{self.namespace}", pod!~".*",name!~".*POD.*"}}[{self.interval}s])) by (pod, src, dst, instance, service) /' \
                    f'sum(rate(out_http_requests_total{{namespace="{self.namespace}", pod!~".*",name!~".*POD.*"}}[{self.interval}s])) by (pod, src,  dst, instance, service) '

        return self.__do_sample__(query)



if __name__ == '__main__':

    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    s = PrometheusSampler(
        podname='daytrader-service',
        interval=1200*0.3334,
        objective='-(throughput / ((((memory / (2**20)) * 0.013375) + (cpu * 0.0535) ) / 2))',
        namespace='default',
        executor=ThreadPoolExecutor(),
        addr='trxrhel7perf-1.canlab.ibm.com',
        port='30099',
        timeout=3600,
        query_range=True,
        start=datetime.fromisoformat('2020-11-26 16:08:00').timestamp(),
        end=datetime.now().timestamp()
    )
    1606401100000
    timeout = 10
    # print(s.cpu().result())
    # print(s.memory().result())
    # print(s.throughput().result())
    # print(s.process_time().result())
    # print(s.error().result())
    # print(s.workload().result(timeout=timeout))
    # print(s.in_out().result(timeout=timeout))
    print(s.metric().objective())