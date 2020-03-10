from common.dataaccess import MongoAccessLayer, PrometheusAccessLayer
from common.timeutil import time_unit, now

import numpy as np
import argparse
import time
import os
import sys


def acmeair_metrics():
    metrics = ['application_com_acmeair_web_LoginRestMetered_login_total',
               'application_com_acmeair_web_LoginRestMetered_logout_total',
               'application_com_acmeair_web_BookingRestMetered_getBookingByNumber_total',
               'application_com_acmeair_web_BookingRestMetered_getBookingsByUser_total',
               'application_com_acmeair_web_BookingRestMetered_bookFlights_total',
               'application_com_acmeair_web_FlightsRestMetered_FlightsREST_total',
               'application_com_acmeair_web_CustomerRestMetered_CustomerREST_total',
               'application_com_acmeair_web_CustomerRestMetered_putCustomer_total',
               'application_com_acmeair_web_BookingRestMetered_BookingsREST_total',
               'application_com_acmeair_web_CustomerRestMetered_getCustomer_total',
               'application_com_acmeair_web_FlightsRestMetered_getTripFlights_total',
               'application_com_acmeair_web_LoginRestMetered_LoginREST_total',
               'application_com_acmeair_web_BookingRestMetered_cancelBookingsByNumber_total']

    return metrics


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

    args = parser.parse_args()

    purl = args.prometheus_url
    pport = args.prometheus_port

    murl = args.mongo_url
    mport = args.mongo_port

    client = PrometheusAccessLayer(purl, pport)

    global MONGO_CLIENT
    MONGO_CLIENT = MongoAccessLayer(murl, mport, args.db_name)
    mongo_collection = MONGO_CLIENT.collection(args.db_collection)
    tunit = time_unit(args.time_unit)
    n = args.fetch_window

    queries = []
    while True:
        _now = now()
        start = _now - tunit(n)
        end = _now

        histogram = {}
        for metric in acmeair_metrics():
            step = tunit(n)
            data = client.increase(metric, start, end, step)
            if not data.is_empty():
                histogram[metric] = data.group(np.sum)
            else:
                histogram[metric] = 0

        if 0 == len(queries) or len(queries) < args.db_buffer_size:
            print(f'buffering query: increase({metric}) start:{start} end:{end} step:{step} hist:{histogram}')
            queries.append({
                'application': 'acmeair',
                'histogram': histogram,
                'start': start,
                'end': end})
        if len(queries) >= args.db_buffer_size:
            print('storing query to db')
            MONGO_CLIENT.store(queries, mongo_collection)
            queries = []

        time.sleep(tunit(n))


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
