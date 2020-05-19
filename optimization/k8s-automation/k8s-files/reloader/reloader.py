import kubernetes as k8s
import time
import os
import traceback

from queue import Queue
from concurrent.futures import ThreadPoolExecutor

class Deployment:
    def __init__(self, namespace, configmap, deployment):
        self.watch = None
        self.namespace = namespace
        self.configmap = configmap
        self.deployment = deployment
        k8s.config.load_incluster_config()

    def patch(self, queue):
        # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/AppsV1Api.md#patch_namespaced_deployment
        body = { "spec": { "template": { "metadata": { "labels": { "date": str(int(time.time())) } } } } }

        api_instance = k8s.client.AppsV1Api(k8s.client.ApiClient())
        pretty = 'true'
        first_loop = True
        while True:
            try:
                event = queue.get()
                name = event["object"].metadata.name
                if not first_loop and 'MODIFIED' == event['type']:
                    print(f'[P] {int(time.time())} Event: type:{event["type"]}, kind:{event["object"].kind}, name:{event["object"].metadata.name}, '
                          f'data:{event["object"].data}')
                    response = api_instance.patch_namespaced_deployment(self.deployment, self.namespace, body, pretty=pretty)
                    print(response)
                first_loop = False
            except:
                traceback.print_exc()

    def watch_config(self, queue):
        api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())
        self.watch = k8s.watch.Watch()
        for event in self.watch.stream(api_instance.list_config_map_for_all_namespaces):
            name = event["object"].metadata.name
            if name == self.configmap:
                queue.put(event)
                print(f'[W] Event: type:{event["type"]}, kind:{event["object"].kind}, name:{name}')
                # print(f'<<< Event: type:{event["type"]}, kind:{event["object"].kind}, name:{name}, '
                #       f'data:{event["object"].data}')

    def close_watch(self):
        print('closing watch configmap')
        if self.watch:
            self.watch.stop()

deployment_name = os.environ.get('DEPLOYMENT_NAME', 'default')
configmap_name = os.environ.get('CONFIGMAP_NAME', '')
namespace = os.environ.get('NAMESPACE', 'default')

executor = ThreadPoolExecutor()

def main():
    queue = Queue()
    deployment = Deployment(namespace, configmap_name, deployment_name)
    try:
        executor.submit(deployment.watch_config, queue)
        executor.submit(deployment.patch, queue)

    except Exception as e:
        traceback.print_exc()
        deployment.close_watch()
        executor.shutdown()

import sys, os
if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     print('Interrupted ', e)
    #     executor.shutdown()
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)
