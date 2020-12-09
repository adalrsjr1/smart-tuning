import unittest
import os
import time

from kubernetes import watch, client, config
from kubernetes.client.models import *
from concurrent.futures import ThreadPoolExecutor
from controllers.k8seventloop import EventLoop, ListToWatch, event_loop
from collections import Counter

import warnings
warnings.simplefilter("ignore", ResourceWarning)

class TestEventLoop(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.executor:ThreadPoolExecutor = ThreadPoolExecutor()
        self.counter = Counter()

    def init(self):
        if 'KUBERNETES_SERVICE_HOST' in os.environ:
            config.load_incluster_config()
        else:
            config.load_kube_config()

    def test_event_loop_intantiation(self):
        loop = EventLoop(executor=self.executor)
        loop.shutdown()

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
        func = list_to_watch.fn()
        r2:V1PodList = func[0](func[1])

        names1 = []
        for item in r1.items:
            names1.append(item.metadata._name)

        names2 = []
        for item in r2.items:
            names2.append(item.metadata._name)

        self.assertListEqual(names1, names2)

    def test_get_fn_wo_args(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespace)
        r1 = v1.list_namespace()
        fn = list_to_watch.fn()
        r2 = fn[0](**fn[1])

        names1 = []
        for item in r1.items:
            names1.append(item.metadata._name)

        names2 = []
        for item in r2.items:
            names2.append(item.metadata._name)

        self.assertListEqual(names1, names2)

    def test_event_loop(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_pod_for_all_namespaces)
        w = watch.Watch()
        f = self.executor.submit(event_loop, w, list_to_watch, lambda x: self.assertEqual(x['object'].kind, 'Pod'))
        time.sleep(1)
        w.stop()
        f.cancel()

    def test_event_loop_size(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_pod_for_all_namespaces)
        w = watch.Watch()

        def counting(event):
            self.counter['test_event_loop_size'] += 1

        f = self.executor.submit(event_loop, w, list_to_watch, counting)
        time.sleep(1)
        w.stop()
        self.assertGreater(self.counter['test_event_loop_size'], 0)
        f.cancel()

    def test_event_loop_w_args(self):
        self.init()
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespaced_pod, namespace='kube-system')
        w = watch.Watch()
        self.executor.submit(event_loop, args=(w, list_to_watch, lambda x: self.assertEqual(x['object']['kind'], 'Pod')))
        time.sleep(1)
        w.stop()

    def test_full_event_loop(self):
        self.init()
        loop = EventLoop(self.executor)

        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_pod_for_all_namespaces)
        loop.register('test', list_to_watch, lambda x: self.assertEqual(x['object'].kind, 'Pod'))
        time.sleep(1)
        loop.shutdown()

    def test_full_event_loop_w_args(self):
        self.init()
        loop = EventLoop(self.executor)

        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespaced_pod, namespace='kube-system')
        loop.register('test', list_to_watch, lambda x: self.assertEqual(x['object'].kind, 'Pod'))
        time.sleep(1)
        loop.shutdown()

    def test_turn_on_off_eventloop(self):
        self.init()
        loop = EventLoop(self.executor)
        v1 = client.CoreV1Api()
        list_to_watch = ListToWatch(v1.list_namespaced_pod, namespace='default')
        loop.register('test', list_to_watch, lambda x: self.assertEqual(x['object'].kind, 'Pod'))
        time.sleep(2)
        loop.unregister('test')
        loop.shutdown()

    @classmethod
    def tearDownClass(self):
        import concurrent
        self.executor.shutdown(wait=False)
        self.executor._threads.clear()
        concurrent.futures.thread._threads_queues.clear()



if __name__ == '__main__':
    unittest.main()
