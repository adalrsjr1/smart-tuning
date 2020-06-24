import logging
import os
import kubernetes as k8s
from kubernetes import client, watch
from kubernetes.client.rest import ApiException
from kubernetes.client.models import *
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

# (key, value) == (production_name, training_name)
duplicated_cm = {}
duplicated_dep = {}
duplicated_svc = {}


def update_service(event):
    if evaluate_event(event):
        try:
            last_applied_service = inject_proxy_into_service(event)
            duplicate_service_for_training(last_applied_service)
        except Exception:
            logger.exception(f'error caused when updating {kind(event["object"])}: {name(event["object"])}')


def evaluate_event(event: dict) -> bool:


    if 'Service' == event['object'].kind and event['type'] in ['MODIFIED']:
        logger.debug(f'will inject proxy into {kind(event["object"])}: {name(event["object"])}?')
        return is_to_inject(event['object'])

    if 'Deployment' == event['object'].kind and event['type'] in ['ADDED']:
        logger.debug(f'will inject proxy into {kind(event["object"])}: {name(event["object"])}?')
        return is_to_inject(event['object'])


    return False


def is_to_inject(k8s_object):
    logger.debug(f'checking "injection.smarttuning.ibm.com" annotation at {kind(k8s_object)}: {name(k8s_object)} ')
    if k8s_object.metadata.annotations:
        annotations = k8s_object.metadata.annotations
        return annotations.get('injection.smarttuning.ibm.com', 'false') == 'true'
    return False


def name(k8s_object):
    if isinstance(k8s_object, dict):
        return k8s_object.get('metadata', {'name': ''}).get('name', '')
    if k8s_object and k8s_object.metadata:
        return k8s_object.metadata.name
    return ''


def kind(k8s_object):
    return k8s_object.kind


def resource_version(k8s_object):
    return k8s_object.metadata.resource_version

def inject_proxy_into_service(event: dict):
    service: V1Service = event['object']
    if is_object_proxied(service):
        logger.info(f'proxy is present at {kind(service)}: {name(service)} -- version {resource_version(service)}')
        return service

    logger.info(f'injecting proxy to {kind(service)}: {name(service)} -- version {resource_version(service)}')
    ports = service.spec.ports

    port = {
        "name": config.PROXY_TAG,
        "port": config.METRICS_PORT,
        "protocol": "TCP",
        "targetPort": config.METRICS_PORT
    }

    ports.append(port)

    patch_body = {
        "kind": service.kind,
        "apiVersion": service.api_version,
        "metadata": {"labels": {"has_proxy": "true", config.PROXY_TAG: "false"}},
        "spec": {
            "selector": {config.PROXY_TAG: 'false'},
            "ports": ports
        }
    }

    try:
        return v1.patch_namespaced_service(name(service), service.metadata.namespace, patch_body)
    except ApiException as e:
        if 409 == e.status:
            logger.exception(f'service {name(service)} conflict')


def is_object_proxied(k8s_object):
    if k8s_object and k8s_object.metadata.labels:
        return k8s_object.metadata.labels.get('has_proxy', 'false') == 'true'


def check_if_is_node_port(service: V1Service):
    return 'NodePort' == service.spec.type


def update_node_port(port_spec, set_value=30999, set_as_none=False):
    value = set_value if not set_as_none else None
    if isinstance(port_spec, dict):
        port_spec.update({'nodePort': value})
    elif isinstance(port_spec, V1ServicePort):
        port_spec.node_port = value

def duplicate_service_for_training(service: V1Service):
    if service is None:
        return None

    if name(service).endswith(f'-{config.PROXY_TAG}'):
        return None

    logger.info(f'dupplicating {kind(service)}: {name(service)}')

    old_name = name(service)
    service.metadata.name += f'-{config.PROXY_TAG}'
    service.metadata.labels.update({config.PROXY_TAG: 'true'})
    service.spec.selector.update({config.PROXY_TAG: 'true'})

    if check_if_is_node_port(service):
        for port in service.spec.ports:
            update_node_port(port, set_as_none=True)

    try:
        v1.create_namespaced_service(namespace=config.NAMESPACE, body=duplicate_service(service))
        duplicated_svc[old_name] = service.metadata.name
    except ApiException as e:
        if 409 == e.status:
            logger.warning(f'deployment {service.metadata.name} is already deployed')

def last_applied_svc_configuration(service:V1Service):
    return json.loads(service.metadata.annotations.get('kubectl.kubernetes.io/last-applied-configuratio', '{}'))

def duplicate_service(service: V1Service):
    service.spec.cluster_ip = None
    service.metadata.resource_version = None

    return service


def update_deployment(event):
    if evaluate_event(event):
        last_applied_deployment = inject_proxy_to_deployment(event)
        duplicate_deployment_for_training(last_applied_deployment)

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

    if is_object_proxied(deployment):
        logger.info(f'proxy is present at {kind(deployment)}: {name(deployment)} -- version {resource_version(deployment)}')
        return deployment

    containers = deployment.spec.template.spec.containers
    first_container: V1Container = containers[0]
    first_port: V1ContainerPort = first_container.ports[0]
    service_port = first_port.container_port

    containers.append(
        proxy_container(proxy_port=config.PROXY_PORT, metrics_port=config.METRICS_PORT, service_port=service_port))
    patch_body = {
        "kind": deployment.kind,
        "apiVersion": deployment.api_version,
        "metadata": {"labels": {"has_proxy": "true", config.PROXY_TAG: "false"}},
        "spec": {
            "template": {
                "metadata": {
                    "labels": {config.PROXY_TAG:"false"}
                },
                "spec": {
                    "initContainers": [init_proxy_container(proxy_port=config.PROXY_PORT, service_port=service_port)],
                    "containers": containers
                }
            },
            # https://github.com/kubernetes/client-go/issues/508
            # "selector": {
            #     "matchLabels": {config.PROXY_TAG:"false"}
            # }
        }
    }

    return v1Apps.patch_namespaced_deployment(deployment.metadata.name, deployment.metadata.namespace, patch_body)


def init_proxy_container(proxy_port: int, service_port: int):
    return {
        'name': 'init-proxy',
        'image': 'smarttuning/init-proxy',
        'imagePullPolicy': 'IfNotPresent',
        'env': [
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


def proxy_container(proxy_port: int, metrics_port: int, service_port: int):
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
            {'name': 'PROXY_PORT', 'value': str(proxy_port)},
            {'name': 'METRICS_PORT', 'value': str(metrics_port)},
            {'name': 'SERVICE_PORT', 'value': str(service_port)},
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
        'name': config.PROXY_NAME,
        'ports': [{'containerPort': proxy_port}, {'containerPort': metrics_port}],
    }


def duplicate_deployment_for_training(deployment: V1Deployment):
    if deployment.metadata.name.endswith(f'-{config.PROXY_TAG}'):
        return None

    old_name = deployment.metadata.name
    deployment.spec.replicas = 1
    deployment.metadata.name += f'-{config.PROXY_TAG}'
    deployment.metadata.labels.update({config.PROXY_TAG: 'true'})
    deployment.spec.template.metadata.labels.update({config.PROXY_TAG: 'true'})
    deployment.spec.selector.match_labels.update({config.PROXY_TAG: 'true'})
    spec: V1PodSpec = deployment.spec.template.spec
    containers = spec.containers

    config_maps = set()
    container: V1Container
    for container in containers:
        if config.PROXY_NAME == container.name:
            continue

        for var in container.env:
            value_from = var.value_from
            if value_from:
                config_map_key_ref = value_from.config_map_key_ref
                if config_map_key_ref:
                    config_maps.add(config_map_key_ref.name)
                    config_map_key_ref.name += f'-{config.PROXY_TAG}'

        for var in container.env_from:
            config_map_ref = var.config_map_ref
            if config_map_ref:
                config_maps.add(config_map_ref.name)

    for volume in spec.volumes:
        config_map = volume.config_map
        if config_map:
            config_maps.add(config_map.name)
            config_map.name += f'-{config.PROXY_TAG}'

    duplication_deployment_configs(config_maps)
    duplicate_deployment(deployment)

    try:
        v1Apps.create_namespaced_deployment(namespace=config.NAMESPACE, body=duplicate_deployment(deployment))
        duplicated_dep[old_name] = deployment.metadata.name
    except ApiException as e:
        if 409 == e.status:
            logger.warning(f'deployment {deployment.metadata.name} is already deployed')


def duplication_deployment_configs(config_maps):
    config_map: V1ConfigMap
    for config_map in v1.list_namespaced_config_map(namespace=config.NAMESPACE).items:
        if config_map.metadata.name in config_maps and not config_map.metadata.name.endswith(f'-{config.PROXY_TAG}'):
            old_name = config_map.metadata.name
            config_map.metadata.name += f'-{config.PROXY_TAG}'
            try:
                v1.create_namespaced_config_map(namespace=config.NAMESPACE, body=duplicate_config_map(config_map))
                duplicated_cm[old_name] = config_map.metadata.name
            except ApiException as e:
                if 409 == e.status:
                    logger.warning(f'config map {config_map.metadata.name} is already deployed')


def duplicate_config_map(config_map: V1ConfigMap):
    return {
        'apiVersion': config_map.api_version,
        'kind': config_map.kind,
        'metadata': {'name': config_map.metadata.name},
        'data': config_map.data
    }


def duplicate_deployment(deployment: V1Deployment):
    metadata: V1ObjectMeta = deployment.metadata
    metadata.resource_version = None
    return deployment


def event_loop(list_to_watch, handler):
    # while True:
    w = watch.Watch()
    for event in w.stream(list_to_watch, config.NAMESPACE):
        logger.info("Event: %s %s %s %s" % (event['type'], kind(event['object']), name(event['object']), resource_version(event['object'])))
        try:
            handler(event)
        except Exception:
            logger.exception('error at event loop')
            w.stop()


## how to do garbage collection on training CMs and DEPs
def init():
    config.executor.submit(event_loop, v1.list_namespaced_service, update_service)
    config.executor.submit(event_loop, v1Apps.list_namespaced_deployment, update_deployment)


if __name__ == '__main__':
    try:
        init()
    except Exception:
        logger.exception('error injector loop')
    finally:
        config.shutdown()
