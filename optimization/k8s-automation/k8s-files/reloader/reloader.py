import kubernetes as k8s
import time

from queue import Queue
from concurrent.futures import ThreadPoolExecutor

class Deployment:
    def __init__(self, namespace, configmap, deployment):
        self.watch = None
        self.queue = Queue()
        self.namespace = namespace
        self.configmap = configmap
        self.deployment = deployment
        k8s.config.load_incluster_config()

    def patch(self):
        # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/AppsV1Api.md#patch_namespaced_deployment
        body = { "spec": { "template": { "metadata": { "labels": { "date": str(int(time.time())) } } } } }

        api_instance = k8s.client.AppsV1Api(k8s.client.ApiClient())
        pretty = 'true'

        while True:
            event = self.queue.get()
            if 'MODIFIED' == event['type']:
                print(f'>>> {int(time.time())} Event: type:{event["type"]}, kind:{event["object"].kind}, name:{event["object"].metadata.name}, '
                      f'data:{event["object"].data}')
                response = api_instance.patch_namespaced_deployment(self.deployment, self.namespace, body, pretty=pretty)
                print(response)

    def watch_config(self):
        api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())
        self.watch = k8s.watch.Watch()
        try:
            for event in self.watch.stream(api_instance.list_config_map_for_all_namespaces):
                name = event["object"].metadata.name
                if name == self.configmap:
                    self.queue.put(event)
                    print(f'<<< Event: type:{event["type"]}, kind:{event["object"].kind}, name:{name}, '
                          f'data:{event["object"].data}')
        except Exception as e:
            print(e)

    def close_watch(self):
        print('closing watch configmap')
        if self.watch:
            self.watch.stop()

def main():
    deployment_name = os.environ.get('DEPLOYMENT_NAME', '')
    configmap_name = os.environ.get('CONFIGMAP_NAME', '')
    namespace = os.environ.get('NAMESPACE', '')

    deployment = Deployment(namespace, configmap_name, deployment_name)
    with ThreadPoolExecutor(2) as executor:
        try:
            executor.submit(deployment.watch_config)
            executor.submit(deployment.patch)

        except Exception as e:
            print(e)
            deployment.close_watch()
            executor.shutdown()

import sys, os
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
