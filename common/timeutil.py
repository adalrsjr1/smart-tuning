import time


def now(past=0):
    return int(time.time()) - past


def second(n):
    return int(n)


def minute(n):
    return second(n * 60)


def hour(n):
    return minute(n * 60)


def day(n):
    return hour(n * 24)


def week(n):
    return day(n * 7)


def month(n):
    return week(n * 4)


def year(n):
    return month(n * 12)


def time_unit(label):
    label = label[0]

    if label == 's':
        return second
    if label == 'm':
        return minute
    if label == 'h':
        return hour
    if label == 'd':
        return day
    if label == 'w':
        return week
    if label == 'M':
        return month
    if label == 'y':
        return year

    return second


def sleep(secs=0):
    time.sleep(secs)
