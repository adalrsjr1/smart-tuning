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


def event_loop(w: watch.Watch, list_to_watch: ListToWatch, callback, context=None):
    def kind(event):
        if isinstance(event['object'], dict):
            return event['object']['kind']
        return event['object'].kind

    def name(event):
        if isinstance(event['object'], dict):
            return event['object']['metadata']['name']
        return event['object'].metadata._name

    loop_name = context[1] if context else ''
    logger.info(f'initializing new watcher loop {loop_name}')

    try:
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
        logger.debug(f'stopping watcher loop {loop_name}')
    except:
        logger.exception('error outside loop ', name(event))


class EventLoop:
    def __init__(self, executor):
        # initializing kubernetes client
        config.init_k8s()
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
        w, f = self.loops[name]
        w.stop()
        f.cancel()
        del self.loops[name]
        return True

    def shutdown(self):

        keys = [key for key in self.loops.keys()]
        for key in keys:
            self.unregister(key)

        logger.info(f'shutdown event loop ')
        w: watch.Watch
        f: Future
        for key, value in self.loops.items():
            logger.info(f'stopping event loop {key}')
            w, f = value
            w.stop()
            f.cancel()
