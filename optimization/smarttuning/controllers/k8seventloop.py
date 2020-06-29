import kubernetes
from kubernetes import watch
from functools import partial
from threading import Thread
import logging
import config
import os

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
        """
        Returns a partial fucntion of the listing function with all parameters wrapped in

        """
        if self.kwargs:
            return partial(self.func, **self.kwargs)
        else:
            return self.func

def event_loop(w:watch.Watch, list_to_watch:ListToWatch, callback, context=None):
    def kind(event):
        if isinstance(event['object'], dict):
            return event['object']['kind']
        return event['object'].kind

    def name(event):
        if isinstance(event['object'], dict):
            return event['object']['metadata']['name']
        return event['object'].metadata.name

    for event in w.stream(list_to_watch.fn()):
        logger.info("Event: %s %s %s" % (event['type'], kind(event), name(event)))
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
    def __init__(self):
        # initializing kubernetes client
        if 'KUBERNETES_SERVICE_HOST' in os.environ:
            kubernetes.config.load_incluster_config()
        else:
            kubernetes.config.load_kube_config()

        self.loops = {}

    def register(self, name, list_to_watch, callback):
        logger.info(f'registering loop "{name}"')
        w = watch.Watch()
        t = Thread(name='name', target=event_loop, args=(w, list_to_watch, callback, (self, name)), daemon=True)
        self.loops[name] = (w, t)
        t.start()

    def unregister(self, name):
        logger.info(f'unregistering loop {name}')
        w, _ = self.loops[name]
        w.stop()
        del self.loops[name]
