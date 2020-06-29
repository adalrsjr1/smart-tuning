import unittest
import os
import time
from kubernetes import watch, client, config
from kubernetes.client.models import *
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from controllers.k8seventloop import EventLoop, ListToWatch, event_loop

import warnings
warnings.simplefilter("ignore", ResourceWarning)

class TestEventLoop(unittest.TestCase):
    executor = ThreadPoolExecutor()

    def init(self):
        if 'KUBERNETES_SERVICE_HOST' in os.environ:
            config.load_incluster_config()
        else:
            config.load_kube_config()

    def test_event_loop_intantiation(self):
        loop = EventLoop()

    def test_list_to_watch(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespaced_pod, namespace='default')
        self.assertDictEqual({'namespace':'default'}, list_to_watch.kwargs)

        list_to_watch = ListToWatch(v1.list_namespaced_pod)
        self.assertDictEqual({}, list_to_watch.kwargs)


    def test_get_fn(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespaced_pod, namespace='default')
        r1:V1PodList = v1.list_namespaced_pod({'namespace':'default'})
        r2:V1PodList = list_to_watch.fn()()

        names1 = []
        for item in r1.items:
            names1.append(item.metadata.name)

        names2 = []
        for item in r2.items:
            names2.append(item.metadata.name)

        self.assertListEqual(names1, names2)

    def test_get_fn_wo_args(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespace)
        r1 = v1.list_namespace()
        r2 = list_to_watch.fn()()

        names1 = []
        for item in r1.items:
            names1.append(item.metadata.name)

        names2 = []
        for item in r2.items:
            names2.append(item.metadata.name)

        self.assertListEqual(names1, names2)

    def test_event_loop(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_pod_for_all_namespaces)
        w = watch.Watch()
        t = Thread(target=event_loop, args=(w, list_to_watch, lambda x: self.assertEqual(x['object'].kind, 'Pod')), daemon=True)
        t.start()
        time.sleep(1)
        w.stop()

    def test_event_loop_w_args(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespaced_pod, namespace='kube-system')
        w = watch.Watch()
        t = Thread(target=event_loop, args=(w, list_to_watch, lambda x: self.assertEqual(x['object']['kind'], 'Pod')), daemon=True)
        t.start()
        time.sleep(1)
        w.stop()

    def test_full_event_loop(self):
        self.init()
        loop = EventLoop()

        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_pod_for_all_namespaces)
        loop.register('test', list_to_watch, lambda x: self.assertEqual(x['object'].kind, 'Pod'))
        time.sleep(1)
        loop.unregister('test')

    def test_full_event_loop_w_args(self):
        self.init()
        loop = EventLoop()

        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespaced_pod, namespace='kube-system')
        loop.register('test', list_to_watch, lambda x: self.assertEqual(x['object']['kind'], 'Pod'))
        time.sleep(1)
        ListToWatch()
        loop.unregister('test')

    def tearDown(self) -> None:
        TestEventLoop.executor.shutdown()




if __name__ == '__main__':
    unittest.main()
