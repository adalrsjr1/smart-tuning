import logging
import os
from concurrent.futures import Future
from functools import partial

import kubernetes
from kubernetes import watch

import config

logger = logging.getLogger(config.EVENT_LOOP_LOGGER)
logger.setLevel(logging.DEBUG)


class ListToWatch:
    def __init__(self, func=None, **kwargs):
        """
        Wrap a kubernetes list type with their paramenters to watch events

        :param func: The k8s client's function that lists a given type
        :param kwargs: The parameters of 'func'
        """
        self.func = func
        self.kwargs = kwargs

    def fn(self):
        return self.func, self.kwargs
        # if self.kwargs:
        #     return self.func, self.kwargs
        # else:
        #     return self.func


def event_loop(w: watch.Watch, list_to_watch: ListToWatch, callback, context=None):
    def kind(event):
        if isinstance(event['object'], dict):
            return event['object']['kind']
        return event['object'].kind

    def name(event):
        if isinstance(event['object'], dict):
            return event['object']['metadata']['name']
        return event['object'].metadata.name

    loop_name = context[1] if context else ''
    logger.info(f'initializing new watching loop {loop_name}')

    for event in w.stream(list_to_watch.func, **list_to_watch.kwargs):
        logger.info("[%s] Event: %s %s %s" % (loop_name, event['type'], kind(event), name(event)))
        try:
            callback(event)
        except Exception:
            logger.exception('error at event loop')
            if context:
                manager = context[0]
                name = context[1]
                manager.unregister(name)
            else:
                w.stop()


class EventLoop:
    def __init__(self, executor):
        # initializing kubernetes client
        if 'KUBERNETES_SERVICE_HOST' in os.environ:
            kubernetes.config.load_incluster_config()
        else:
            kubernetes.config.load_kube_config()
        self.executor = executor
        self.loops = {}

    def register(self, name, list_to_watch, callback) -> (str, watch.Watch, Future):
        logger.info(f'registering loop "{name}"')
        w = watch.Watch()
        f = self.executor.submit(event_loop, w, list_to_watch, callback, (self, name))
        self.loops[name] = (w, f)
        return name, w, f

    def unregister(self, name):
        if not name in self.loops:
            logger.info(f'loop {name} not registered')
            return False
        logger.info(f'unregistering loop {name}')
        w, _ = self.loops[name]
        w.stop()
        del self.loops[name]
        return True

    def shutdown(self):
        logger.info(f'shutdown event loop ')
        w: watch.Watch
        f: Future
        for key, value in self.loops.items():
            logger.info(f'stopping event loop {key}')
            w, f = value
            w.stop()
            f.cancel()
