import random
import time

from prometheus_client import start_http_server, Info, Gauge, Counter

start_http_server(8000)
instruments = {}


def labels_factory(app: str, namespace: str, iteration_name: str, iteration_counter: int, training: bool, workload: str,
                   pruned: bool):
    return {
        'app': app,
        'namespace': namespace,
        'iteration_name': iteration_name,
        'iteration_counter': iteration_counter,
        'training': training,
        'workload': workload,
        'pruned': pruned,
    }


def info_metric(labels: dict, name, description: str, infos: dict):
    if name not in instruments:
        instruments[id] = Info(name, description, list(labels.keys()))
    i: Info = instruments[id]
    i.labels(**labels).info(infos)


def gauge_metric(labels: dict, name: str, description: str, value: float):
    if name not in instruments:
        instruments[name] = Gauge(name, description, list(labels.keys()))
    g: Gauge = instruments[name]
    g.labels(**labels).set(value)


def counter_metric(labels: dict, name: str, description: str, value=1):
    if name not in instruments:
        instruments[name] = Counter(name, description, list(labels.keys()))
    c: Counter = instruments[name]
    c.labels(**labels).inc(value)


def foo(id, name, value, labels):
    if not id in instruments:
        instruments[id] = Info(id, 'foo_metric', list(labels.keys()))
    i: Info = instruments[id]
    i.labels(**labels).info({'test1': str(name), 'test2': str(value)})


def foox(id, name, value, labels):
    if not id in instruments:
        instruments[id] = Info(id, 'foo_metric', list(labels.keys()))
    i: Info = instruments[id]
    i.labels(**labels).info({'x1': str(name), 'x2': str(value)})


def bar(id, value, labels):
    if id not in instruments:
        instruments[id] = Gauge(id, 'foo_metric', list(labels.keys()))
    g: Gauge = instruments[id]
    g.labels(**labels).set(value)


if __name__ == '__main__':
    foo('foo_metric', 'foo', 3.14, {'a': '1', 'b': True})
    foo('foo_metric2', 'foo', 1, {'a': '1', 'b': True})
    foo('foo_metric', 'foo', 3.14, {'a': '1', 'b': True})
    foox('foo_metric', 'foo', 3.14, {'a': '2', 'b': True})
    foo('foo_metric', 'foo', 3.14, {'a': '1', 'b': True})
    foo('foo_metric', 'foo', 3.14, {'a': '2', 'b': True})
    foo('foo_metric', 'foo', 3.14, {'a': '1', 'b': True})
    foox('foo_metric', 'foo', 3.14, {'a': '1', 'b': True})



    for i in range(100):
        time.sleep(1)
        bar('bar_metric', random.randint(-100, 100), {'a': '1', 'b': True, 'c': i})

    time.sleep(3600)
