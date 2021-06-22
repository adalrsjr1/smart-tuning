import random
import time

from prometheus_client import Info, Gauge, Counter


__instruments = {}


def info_metric(labels: dict, name, description: str, infos: dict):
    if name not in __instruments:
        __instruments[id] = Info(name, description, list(labels.keys()))
    i: Info = __instruments[id]
    i.labels(**labels).info(infos)


def gauge_metric(labels: dict, name: str, description: str, value: float):
    if name not in __instruments:
        __instruments[name] = Gauge(name, description, list(labels.keys()))
    g: Gauge = __instruments[name]
    g.labels(**labels).set(value)


def counter_metric(labels: dict, name: str, description: str, value=1):
    if name not in __instruments:
        __instruments[name] = Counter(name, description, list(labels.keys()))
    c: Counter = __instruments[name]
    c.labels(**labels).inc(value)
