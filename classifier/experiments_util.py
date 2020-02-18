import numpy as np

from classifier import workload_comparision as wc
from common.dataaccess import PrometheusAccessLayer, PrometheusResponse


def statistics(prometheus: PrometheusAccessLayer, start: int, end: int, step: int):
    avg = prometheus.avg('base_memory_usedHeap_bytes', start, end, step).group(np.mean)
    std = prometheus.std('base_memory_usedHeap_bytes', start, end, step).group(np.mean)

    expressions = ["application_com_acmeair_web_LoginRestMetered_LoginREST_total",
                   "application_com_acmeair_web_LoginRestMetered_login_total",
                   "application_com_acmeair_web_LoginRestMetered_logout_total",
                   "application_com_acmeair_web_BookingRestMetered_BookingsREST_total",
                   "application_com_acmeair_web_BookingRestMetered_getBookingByNumber_total",
                   "application_com_acmeair_web_BookingRestMetered_bookFlights_total",
                   "application_com_acmeair_web_BookingRestMetered_getBookingsByUser_total",
                   "application_com_acmeair_web_BookingRestMetered_cancelBookingsByNumber_total",
                   "application_com_acmeair_web_FlightsRestMetered_FlightsREST_total",
                   "application_com_acmeair_web_FlightsRestMetered_getTripFlights_total",
                   "application_com_acmeair_web_CustomerRestMetered_CustomerREST_total",
                   "application_com_acmeair_web_CustomerRestMetered_putCustomer_total",
                   "application_com_acmeair_web_CustomerRestMetered_getCustomer_total"]

    hist = [prometheus.increase(expression, start, end, step).group(np.mean) for expression in expressions]
    # print(hist)
    # print(f'avg:{avg}\nstd: {std}\nhist: {hist}\ndff: {end - start}\n')
    return avg, std, np.array(hist), (end - start) / step, end - start


if __name__ == '__main__':
    prometheus = PrometheusAccessLayer('localhost', 9090)

    computed = [statistics(prometheus, 1582037906, 1582038209, 300),  # 0.0
                statistics(prometheus, 1582036783, 1582037073, 300),  # 0.3
                statistics(prometheus, 1582037162, 1582037435, 300),  # 0.7
                statistics(prometheus, 1582039185, 1582039500, 300)]  # 1.0

    labels = [0.0, 0.3, 0.7, 1.0]
    # for label in labels:
    #     print(f'{label:6}', end=' ')
    # print()
    #
    # for label in labels:
    #     print(f'{"-------|":6}', end=' ')
    # print()

    for stat1 in computed:
        for stat2 in computed:
            value = wc.ztest(stat1[0], stat2[0], stat1[1], stat2[1], 300 / 15)
            print(f'{value:6.4f},', end=' ')
        print()
    print()
    #
    # for label in labels:
    #     print(f'{label:6}', end=' ')
    # print()
    #
    # for label in labels:
    #     print(f'{"-------|":6}', end=' ')
    # print()
    for stat1 in computed:
        for stat2 in computed:
            value = wc.pearson(stat1[2], stat2[2])
            print(f'{value:6.4},', end=' ')
        print()
