from common.dataaccess import MongoAccessLayer, PrometheusAccessLayer, PrometheusResponse
from common.timeutil import time_unit, now

import numpy as np
import argparse
import time
import os
import sys

MONGO_CLIENT = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prometheus-url', type=str, required=True)
    parser.add_argument('--prometheus-port', type=int, required=True)
    parser.add_argument('--mongo-url', type=str, required=True)
    parser.add_argument('--mongo-port', type=int, required=True)

    parser.add_argument('--fetch_window', help='Time interval between each Prometheus query', type=int, required=True)
    parser.add_argument('--time_unit',
                        help='Time unit for the time interval [s]econd, [m]inute, [h]our, [d]ay, [w]eek, [M]onth, [y]ear',
                        type=str, required=True)

    parser.add_argument('--db-name', help='Mongodb name', type=str, required=True)
    parser.add_argument('--db-collection', help='Mongodb collection name', type=str, required=True)

    parser.add_argument('--db-buffer-size', help='Size buffer before store data', type=int, default=0)

    parser.add_argument('--metric-name', help='Name of metric to be add to db', type=str, required=True)
    parser.add_argument('--metric-query', help='Prometheus\' query to fetch metric data', type=str, required=True)

    args = parser.parse_args()
    purl = args.prometheus_url
    pport = args.prometheus_port

    murl = args.mongo_url
    mport = args.mongo_port

    client = PrometheusAccessLayer(purl, pport)
    global MONGO_CLIENT
    global MONGO_CLIENT
    MONGO_CLIENT = MongoAccessLayer(murl, mport, args.db_name)
    mongo_collection = MONGO_CLIENT.collection(args.db_collection)
    tunit = time_unit(args.time_unit)
    n = args.fetch_window

    queries = []
    query = args.metric_query
    metric = args.metric_name
    while True:
        _now = now()
        start = _now - tunit(n)
        end = _now

        step = tunit(n)
        # query = f'(base_memory_usedHeap_bytes/base_memory_maxHeap_bytes)'
        # query = f'base_cpu_processCpuLoad_percent'
        # query = f'vendor_threadpool_size{pool="LargeThreadPool"'
        # query = f'increase(application_currentHttpConnections[10s])'
        # query = f'increase({metric}[{step}s)'
        # query = rate(vendor_servlet_request_total{servlet="acmeair_webapp_com_acmeair_web_AcmeAirApp"}[step])
        # query = rate(vendor_servlet_responseTime_total_seconds{servlet="[[servlet]]"}[$__interval])/rate(vendor_servlet_request_total{servlet="[[servlet]]"}[$__interval])
        data = client.query(query, start, end, step)
        value = data.group(np.sum)

        if len(queries) == 0 or len(queries) < args.db_buffer_size:
            print(f'buffering query:{query} metric:{metric} start:{start} value:{value} end:{end} step:{step}')
            queries.append({
                'application': 'acmeair',
                'metric': metric,
                'value': value,
                'start': start,
                'end': end})
        if len(queries) >= args.db_buffer_size:
            print('storing query to db')
            MONGO_CLIENT.store(queries, mongo_collection)
            queries = []

        time.sleep(n)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        MONGO_CLIENT.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
