import numpy as np
import math


def norm(u: np.ndarray) -> np.ndarray:
    _min = u.min()
    _max = u.max()

    if (_max - _min) == 0:
        return np.zeros(len(u))

    return (u - _min) / (_max - _min)


def ztest(mean1, mean2, std1, std2) -> float:
    # http://homework.uoregon.edu/pub/class/es202/ztest.html
    """ If the Z-statistic is less than 2, the two samples are the same.
        If the Z-statistic is between 2.0 and 2.5, the two samples are marginally different
        If the Z-statistic is between 2.5 and 3.0, the two samples are significantly different
        If the Z-statistic is more then 3.0, the two samples are highly signficantly different """
    return abs((mean1 - mean2) / math.sqrt(std1 ** 2 + std2 ** 2))


def check_size(u: np.ndarray, v: np.ndarray):
    if not len(u) == len(v):
        raise ValueError(f'u and v must be of the same size')


def hellinger(u: np.ndarray, v: np.ndarray) -> float:
    """ If returns 0.0 matchs
        if returns 1.0 mismatch"""

    check_size(u, v)

    u = norm(u)
    v = norm(v)

    a = np.sum(np.sqrt(u * v))
    b = np.sqrt(np.mean(u) * np.mean(v) * len(u) ** 2)
    #    b = 1 / b
    b = np.divide(1, b, out=np.zeros_like(a), where=b != 0)
    # assert (a * b) <= 1, f'{a*b}'

    return math.sqrt((1 - (round(a * b, 2))))


def pearson(u: np.ndarray, v: np.ndarray) -> float:
    """ If returns 1.0 matchs
        If returns 0.5 half matchs
        if returns 0.0 mismatch"""

    check_size(u, v)

    u = norm(u)
    v = norm(v)

    _u = np.mean(u)
    _v = np.mean(v)

    a = np.sum((u - _u) * (v - _v))
    b = np.sqrt(np.sum((u - _u) ** 2) * np.sum((v - _v) ** 2))

    p = a / b

    # return p - (-1) / (1 - (-1))
    return p


def chisquare(u: np.ndarray, v: np.ndarray) -> float:
    """ If returns 0.0 matchs
        if returns unbounded mismatch"""

    check_size(u, v)

    u = norm(u)
    v = norm(v)

    return np.sum((u - v) ** 2 // u)


def chisquare_alt(u: np.ndarray, v: np.ndarray) -> float:
    """ If returns 0.0 match
        unbounded mismatch """

    check_size(u, v)

    u = norm(u)
    v = norm(v)

    return 2 * np.sum(((u - v) ** 2) // (u + v))


def kullback(u: np.ndarray, v: np.ndarray) -> float:
    check_size(u, v)

    # u = norm(u)
    # v = norm(v)

    return np.sum(u * np.log(u / v))


def intersection(u: np.ndarray, v: np.ndarray) -> float:
    check_size(u, v)

    u = norm(u)
    v = norm(v)

    return np.sum(np.minimum(u, v))


if __name__ == '__main__':
    mu_1 = -6  # mean of the first distribution
    mu_2 = 6  # mean of the second distribution
    data_1 = np.random.normal(mu_1, 2.0, 1000)
    data_2 = np.random.normal(mu_2, 2.0, 1000)
    hist_1, _ = np.histogram(data_1, bins=100, range=[-15, 15])
    hist_2, _ = np.histogram(data_2, bins=100, range=[-15, 15])

    data_1 = hist_1
    data_2 = hist_2

    ztest1 = ztest(np.mean(data_1), np.mean(data_2), np.std(data_1), np.std(data_2))
    print(f'ztest: {ztest1}')
    print(f'helli: {hellinger(data_1, data_2)}')
    print(f'pears: {pearson(data_1, data_2)}')
    print(f'chisq: {chisquare(data_1, data_2)}')
    print(f'chial: {chisquare_alt(data_1, data_2)}')
    print(f'kullb: {kullback(data_1, data_2)}')
    print(f'inter: {intersection(data_1, data_2)}')
