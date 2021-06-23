import random
import threading
import time
from numbers import Number
from typing import Optional

import prometheus_client
from prometheus_client import Info, Gauge, Counter, Enum

__collectors = {}


def info_metric(labels: dict, name, description: str, infos: dict):
    if name not in __collectors:
        __collectors[id] = Info(name, description, list(labels.keys()))
    i: Info = __collectors[id]
    i.labels(**labels).info(infos)


def gauge_metric(labels: dict, name: str, description: str, value: Optional[Number] = None):
    if name not in __collectors:
        __collectors[name] = Gauge(name, description, list(labels.keys()))
    g: Gauge = __collectors[name]
    if value is None:
        g.labels(**labels).inc(1)
    else:
        g.labels(**labels).set(value)


def counter_metric(labels: dict, name: str, description: str, value=1):
    if name not in __collectors:
        __collectors[name] = Counter(name, description, list(labels.keys()))
    c: Counter = __collectors[name]
    c.labels(**labels).inc(value)


def enum_metric(labels: dict, name: str, description: str, states: list[str], state):
    if name not in __collectors:
        __collectors[name] = Enum(name, description, list(labels.keys()), states=states)
    e: Enum = __collectors[name]
    e.labels(**labels).state(state)


def __new_coolectors(collectors: dict) -> dict:
    return {name: collector for name, collector in collectors.items() if isinstance(collector, Counter)}