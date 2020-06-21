import logging
import os
import kubernetes as k8s
from kubernetes import client, watch
from kubernetes.client.models import V1Deployment, V1Container, V1ContainerPort, V1Service
import json
import config

if 'KUBERNETES_SERVICE_HOST' in os.environ:
    k8s.config.load_incluster_config()
else:
    k8s.config.load_kube_config()

logger = logging.getLogger('smarttuning_injector')
logger.setLevel(logging.DEBUG)

v1 = client.CoreV1Api()
v1Apps = client.AppsV1Api()

def update_service(event):
    if is_to_inject(event):
        inject_proxy_to_service(event)

def inject_proxy_to_service(event: dict):
    service:V1Service = event['object']

    ports = service.spec.ports
    nodePort = {'nodePort': 30090}
    port = {
        "name": "smarttuning",
        "port": config.METRICS_PORT,
        "protocol": "TCP",
        "targetPort": config.METRICS_PORT
    }

    if 'NodePort' == service.spec.type:
        port.update(nodePort)

    ports.append(port)

    patch_body = {
        "kind": service.kind,
        "apiVersion": service.api_version,
        "metadata": {"labels": {"has_proxy": "true"}},
        "spec": {"ports": ports}
    }

    return v1.patch_namespaced_service(service.metadata.name, service.metadata.namespace, patch_body)



def update_deployment(event):

    if is_to_inject(event):
        inject_proxy_to_deployment(event)

def is_to_inject(event: dict) -> bool:
    if 'Service' == event['object'].kind and not event['type'] in ['MODIFIED']:
        return False

    if 'Deployment' == event['object'].kind and not event['type'] in ['ADDED', 'MODIFIED']:
        return False

    # fetch from this annotation to avoid race condition
    annotations = extracts_annotations(event)
    raw_object = annotations.get('kubectl.kubernetes.io/last-applied-configuration', {})
    if not raw_object:
        return False
    parsed_object = json.loads(raw_object)
    annotations: dict = parsed_object['metadata']['annotations']
    to_inject = annotations.get('injection.smarttuning.ibm.com', 'false').lower()
    revision = int(annotations.get('service.smarttuning.ibm.com/revision', '1'))
    return 'true' == to_inject and 1 >= revision


def extracts_annotations(event):
    annotations = event['object'].metadata.annotations or {}
    return annotations

def inject_proxy_to_deployment(event):
    deployment: V1Deployment = event['object']
    containers = deployment.spec.template.spec.containers
    first_container:V1Container = containers[0]
    first_port:V1ContainerPort = first_container.ports[0]
    service_port = first_port.container_port

    containers.append(proxy_container(proxy_port=config.PROXY_PORT, metrics_port=config.METRICS_PORT, service_port=service_port))
    patch_body = {
        "kind": deployment.kind,
        "apiVersion": deployment.api_version,
        "metadata": {"labels": {"has_proxy": "true"}},
        "spec": {"template": {"spec": {
            "initContainers": [init_proxy_container(proxy_port=config.PROXY_PORT, service_port=service_port)],
            "containers": containers
        }}}
    }

    return v1Apps.patch_namespaced_deployment(deployment.metadata.name, deployment.metadata.namespace, patch_body)

def init_proxy_container(proxy_port:int, service_port:int):
    return {
        'name': 'init-proxy',
        'image': 'smarttuning/init-proxy',
        'imagePullPolicy': 'IfNotPresent',
        'env':[
            {
                'name': 'PROXY_PORT',
                'value': str(proxy_port)
            },
            {
                'name': 'SERVICE_PORT',
                'value': str(service_port)
            }
        ],
        'securityContext': {
            'capabilities': {
                'add': [
                    'NET_ADMIN'
                ]
            }
        },
        'privileged': 'true'
    }


def proxy_container(proxy_port:int, metrics_port:int, service_port:int):
    env_var = lambda name, path: {
              'name': name,
              'valueFrom': {
                  'fieldRef': {
                      'fieldPath': path
                  }
              }
          }
    return {
      'env': [
          {'name':'PROXY_PORT', 'value':str(proxy_port)},
          {'name':'METRICS_PORT', 'value':str(metrics_port)},
          {'name':'SERVICE_PORT', 'value':str(service_port)},
          env_var('NODE_NAME', 'spec.nodeName'),
          env_var('POD_NAME', 'metadata.name'),
          env_var('POD_NAMESPACE', 'metadata.namespace'),
          env_var('POD_IP', 'status.podIP'),
          env_var('POD_SERVICE_ACCOUNT', 'spec.serviceAccountName'),
      ],
      'envFrom': [{
          'configMapRef': {
              'name': 'smarttuning-proxy-config'
          }
      }],
      'image': 'smarttuning/proxy',
      'imagePullPolicy': 'IfNotPresent',
      'name': 'proxy',
      'ports': [{'containerPort': proxy_port}, {'containerPort': metrics_port}],
      }

def event_loop(list_to_watch, handler):
    while True:
        w = watch.Watch()
        for event in w.stream(list_to_watch, config.NAMESPACE):
            logger.info("Event: %s %s %s" % (event['type'], event['object'].kind, event['object'].metadata.name))
            try:
                handler(event)
            except Exception:
                w.stop()

def init():
    config.executor.submit(event_loop, v1.list_namespaced_service, update_service)
    config.executor.submit(event_loop, v1Apps.list_namespaced_deployment, update_deployment)

if __name__ == '__main__':
    try:
        init()
    except Exception:
        logger.exception()
    finally:
        config.shutdown()