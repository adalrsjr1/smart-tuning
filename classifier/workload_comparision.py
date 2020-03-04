import numpy as np
import math


def norm(u: np.ndarray) -> np.ndarray:
    u = np.array(u)
    _min = u.min()
    _max = u.max()

    return (u - _min) / (_max - _min)


def norm_max(u: np.ndarray) -> np.ndarray:
    return u / np.finfo(np.float).max


def ztest(mean1, mean2, std1, std2, n_data_points) -> float:
    # http://homework.uoregon.edu/pub/class/es202/ztest.html
    """ If the Z-statistic is less than 2, the two samples are the same.
        If the Z-statistic is between 2.0 and 2.5, the two samples are marginally different
        If the Z-statistic is between 2.5 and 3.0, the two samples are significantly different
        If the Z-statistic is more then 3.0, the two samples are highly signficantly different """
    sigma1 = std1 / math.sqrt(n_data_points)
    sigma2 = std2 / math.sqrt(n_data_points)

    return abs((mean1 - mean2) / math.sqrt(sigma1 ** 2 + sigma2 ** 2))


def check_size(u: np.ndarray, v: np.ndarray):
    if len(u) != len(v):
        raise ValueError(f'u and v must be of the same size')


def hellinger(u: np.ndarray, v: np.ndarray) -> float:
    """ If returns 0.0 matchs
        if returns 1.0 mismatch"""

    check_size(u, v)

    u = norm(u)
    v = norm(v)

    return np.sqrt(np.sum((np.sqrt(u) - np.sqrt(v)) ** 2)) / np.sqrt(2)


def hellinger2(u: np.ndarray, v: np.ndarray) -> float:
    check_size(u, v)

    u1 = norm([u, v])[0]
    v1 = norm([u, v])[1]

    print(u1)
    print(v1)

    u = u1
    v = v1

    return np.sqrt(np.sum((np.sqrt(u) - np.sqrt(v)) ** 2)) / np.sqrt(2)


def pearson(u: np.ndarray, v: np.ndarray) -> float:
    """ If returns 1.0 matchs
        If returns 0.5 half matchs
        if returns 0.0 mismatch"""

    check_size(u, v)

    u1 = norm([u, v])[0]
    v1 = norm([u, v])[1]

    _u = np.mean(u)
    _v = np.mean(v)

    a = np.sum((u - _u) * (v - _v))
    b = np.sqrt(np.sum((u - _u) ** 2) * np.sum((v - _v) ** 2))

    p = a / b

    # return p - (-1) / (1 - (-1))
    return p

def pearson2(u: np.ndarray, v: np.ndarray) -> float:
    """ If returns 1.0 matchs
        If returns 0.5 half matchs
        if returns 0.0 mismatch"""

    check_size(u, v)

    u1 = norm([u,v])[0]
    v1 = norm([u,v])[1]

    _u = np.mean(u1)
    _v = np.mean(v1)

    a = np.sum((u1 - _u) * (v1 - _v))
    b = np.sqrt(np.sum((u1 - _u) ** 2) * np.sum((v1 - _v) ** 2))

    return a/b

def chisquare(u: np.ndarray, v: np.ndarray) -> float:
    """ If returns 0.0 matchs
        if returns unbounded mismatch"""

    check_size(u, v)

    return np.sum(((u - v) ** 2) / (u + v))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    check_size(u, v)

    return 1 - np.sum(u * v)

def kullback(u: np.ndarray, v: np.ndarray) -> float:
    check_size(u, v)

    return np.sum(u * np.log(u / v))


def intersection(u: np.ndarray, v: np.ndarray) -> float:
    check_size(u, v)

    u = norm(u)
    v = norm(v)

    return np.sum(np.minimum(u, v))


if __name__ == '__main__':
    mu_1 = 0.5  # mean of the first distribution
    mu_2 = 0.5  # mean of the second distribution
    data_1 = np.random.normal(mu_1, 0., 1000)
    data_2 = np.random.normal(mu_2, 0., 1000)
    hist_1, _ = np.histogram(data_1, bins=100, range=[-15, 15])
    hist_2, _ = np.histogram(data_2, bins=100, range=[-15, 15])

    data_1 = hist_1
    data_2 = hist_2

    ztest1 = ztest(76.4, 102, 12.97, 4.08, 300 // 15)

    # print(f'ztest: {ztest1}')
    print(f'helli: {hellinger(data_1, data_2)}')
    print(f'pears: {pearson(data_1, data_2)}')
    # print(f'chisq: {chisquare(data_1, data_2)}')
    # print(f'chial: {chisquare_alt(data_1, data_2)}')
    # print(f'kullb: {kullback(data_1, data_2)}')
    # print(f'inter: {intersection(data_1, data_2)}')
