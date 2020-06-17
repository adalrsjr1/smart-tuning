from __future__ import annotations
from concurrent.futures import Future
import config
import logging
import timeinterval

from prometheus_pandas import query as handler

logging.debug(f'connecting to prometheus at {config.PROMETHEUS_ADDR}:{config.PROMETHEUS_PORT}')
_prometheus = handler.Prometheus(f'http://{config.PROMETHEUS_ADDR}:{config.PROMETHEUS_PORT}')

def do_sample(query:str, endpoint=_prometheus) -> Future:
    return config.executor.submit(endpoint.query, query)

def throughput(pod_regex, interval, quantile=1.0, endpoint=_prometheus) -> Future:
    """ return a concurrent.futures.Future<pandas.Series> with the total_requests rate of an specific pod"""
    logging.debug(f'sampling throughput at {pod_regex}')
    query = f'quantile({quantile},rate(smarttuning_http_requests_total{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s]))'

    return do_sample(query, endpoint=endpoint)

def latency(pod_regex, interval, quantile=1.0, endpoint=_prometheus) -> Future:
    """ return a concurrent.futures.Future<pandas.Series> with the latency_sum/latency_count rate of an specific pod"""
    logging.debug(f'sampling latency at {pod_regex}')
    query = f'quantile({quantile}, rate(smarttuning_http_latency_seconds_sum{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s])) / ' \
    f'quantile({quantile}, rate(smarttuning_http_latency_seconds_count{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s]))'

    return do_sample(query, endpoint=endpoint)

def memory(pod_regex, interval, quantile=1.0, endpoint=_prometheus) -> Future:
    """ return a concurrent.futures.Future<pandas.Series> with the memory (bytes) quantile over time of an specific pod
        :param quantile a value 0.0 - 1.0
    """
    logging.debug(f'sampling memory at {pod_regex}')
    query = f'quantile({quantile}, quantile_over_time({quantile},container_memory_working_set_bytes{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s]))'

    return do_sample(query, endpoint=endpoint)

def cpu(pod_regex, interval, quantile=1.0, endpoint=_prometheus) -> Future:
    """ return a concurrent.futures.Future<pandas.Series> with the CPU (milicores) rate over time of an specific pod
    """
    logging.debug(f'sampling cpu at {pod_regex}')
    query = f'quantile({quantile},rate(container_cpu_usage_seconds_total{{pod=~"{pod_regex}",name!~".*POD.*"}}[{timeinterval.second(interval)}s]))'
    return do_sample(query, endpoint=endpoint)

def workload(pod_regex, interval, endpoint=_prometheus) -> Future:
    """
    query urls requests distribution in a given interval
    :param pod_regex: application namespace
    :param interval: query for throughput in the past interval (uses module 'timeinterval')
    :return: Workload object. Note that urls are grouped in case of using <Path Parameters>, e.g.,

    the urls:

    /my/url/using/path-parameter/uid153@email.com
    /my/url/using/path-parameter/uid188@email.com
    /my/url/using/path-parameter/uid174@email.com

    are grouped into /my/url/using/path-parameter/uid153@email.com

    """
    logging.debug(f'sampling urls at {pod_regex}')

    query = f'sum by (path)(rate(smarttuning_http_requests_total{{pod=~"{pod_regex}"}}[{timeinterval.second(interval)}s]))' \
            f' / ignoring ' \
            f'(path) group_left sum(rate(smarttuning_http_requests_total{{pod=~"{pod_regex}"}}[{timeinterval.second(interval)}s]))'

    return do_sample(query, endpoint=endpoint)
