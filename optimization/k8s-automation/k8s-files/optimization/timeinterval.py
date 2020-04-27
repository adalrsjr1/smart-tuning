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


def time_unit(label: str):
    label = label[0]

    if label.lower() == 's':
        return second
    if label.lower() == 'm':
        return minute
    if label.lower() == 'h':
        return hour
    if label.lower() == 'd':
        return day
    if label.lower() == 'w':
        return week
    if label.lower() == 'M':
        return month
    if label.lower() == 'y':
        return year

    return second


def sleep(secs=0):
    time.sleep(secs)