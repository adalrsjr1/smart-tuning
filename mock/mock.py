import matplotlib.pyplot as plt
import numpy as np


def sinusoidal1(t, mean=0, stddev=0):
    return np.abs(
        np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * 2.25 * 2 * np.pi * t) + np.random.normal(mean, stddev, len(t)))


def sinusoidal2(t, mean=0, stddev=0):
    return np.abs(np.sin(2 * np.pi * t))


def sinusoidal3(t, mean=0, stddev=0):
    return np.sin(2 * 2.25 * 2 + np.pi * t)


def sub_sinusoidal(which_pattern, n_points, mean=0, stddev=0):
    """
    pick one of the n patterns in the curve
    :param which_pattern:
    :param n_points:
    :param mean: noise frequency
    :param stddev: noise amplitude
    :return: sinusoidal values and time interval
    """
    subintervals = np.linspace(0, 2, 5)

    assert len(subintervals) >= which_pattern >= 0

    i0 = subintervals[which_pattern]
    i1 = subintervals[which_pattern + 1]

    t = np.linspace(i0, i1, n_points)
    s = sinusoidal1(t, mean, stddev)

    return s, t


def pick_one_pattern(t):
    if t % 7 == 0:
        return 0
    if t % 13 == 1:
        return 1
    if t % 17 == 2:
        return 2
    if t % 19 == 3:
        return 3

    return -1


def pick_next_pattern(given_p):
    pattern = -1
    while pattern < 0:
        pattern = pick_one_pattern(given_p)
        given_p += 1

    return pattern, given_p


def time_series(size, mutation_probability=0.0):
    serie = []

    count = 0
    while len(serie) < size:
        pattern, count = pick_next_pattern(count)
        r = np.random.rand()
        if r < mutation_probability:
            pattern = (pattern + np.random.randint(4) + 1) % 4
        serie.append(pattern)

        count += 1
    return serie


def periodic_time_series(n_periods, period_lenght, mutation_probability=0.0):
    values = []
    for i in range(n_periods):
        values.append(time_series(period_lenght, mutation_probability=mutation_probability))

    return [item for sublist in values for item in sublist]


def plot(x, y):
    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    for i in range(10):
        print(time_series(15, 0.0))
        # x = pick_one_pattern(i)
        #
        # x = x if x >= 0 else 0
        #
        # v, t = sub_sinusoidal(x, 50, mean=0, stddev=0)
        #
        # plot(t, v)
