from __future__ import annotations
import json
import logging
import os
import time

import kubernetes as k8s
from kubernetes import client, watch
from kubernetes.client.models import *
from kubernetes.client.rest import ApiException

import config
from controllers.k8seventloop import EventLoop, ListToWatch

config.init_k8s(hostname=config.K8S_HOST)

logger = logging.getLogger(config.INJECTOR_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)

# (key, value) == (production_name, training_name)
duplicated_cm = {}
duplicated_dep = {}
duplicated_svc = {}

def training_suffix() -> str:
    return f'{config.PROXY_TAG}'

def update_service(event):
    if evaluate_event(event):
        try:
            last_applied_service = inject_proxy_into_service(event)
            duplicate_service_for_training(last_applied_service)
        except Exception:
            logger.exception(f'error caused when updating {kind(event["object"])}: {k8s_object_name(event["object"])}')


def evaluate_event(event: dict) -> bool:
    if 'Service' == event['object'].kind and event['type'] in ['ADDED', 'MODIFIED']:
        logger.debug(f'will inject proxy into {kind(event["object"])}: {k8s_object_name(event["object"])}?')
        return is_to_inject(event['object'])

    if 'Deployment' == event['object'].kind and event['type'] in ['ADDED']:
        logger.debug(f'will inject proxy into {kind(event["object"])}: {k8s_object_name(event["object"])}?')
        return is_to_inject(event['object'])

    return False


def is_to_inject(k8s_object):
    logger.debug(f'checking "injection.smarttuning.ibm.com" annotation at {kind(k8s_object)}: {k8s_object_name(k8s_object)} ')
    if k8s_object.metadata.annotations:
        annotations = k8s_object.metadata.annotations
        return annotations.get('injection.smarttuning.ibm.com', 'false') == 'true'
    return False


def k8s_object_name(k8s_object):
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
        logger.info(f'proxy is present at {kind(service)}: {k8s_object_name(service)} -- version {resource_version(service)}')
        return service

    logger.info(f'injecting proxy to {kind(service)}: {k8s_object_name(service)} -- version {resource_version(service)}')
    ports = service.spec.ports



    port = {
        "name": config.PROXY_TAG,
        "port": config.METRICS_PORT,
        "protocol": "TCP",
        "targetPort": config.METRICS_PORT
    }

    ports.append(port)
    config.coreApi().patch_namespaced_pod
    patch_body = {
        "kind": service.kind,
        "apiVersion": service.api_version,
        "metadata": {"labels": {"has_proxy": "true", config.PROXY_TAG: "false"}},
        "spec": {
            "selector": {config.PROXY_TAG: 'false'},
            "ports": ports
        }
    }

    config.init_k8s(config.K8S_HOST)
    try:

        return config.coreApi().patch_namespaced_service(k8s_object_name(service), service.metadata.namespace, patch_body)
    except ApiException as e:
        if 409 == e.status:
            logger.exception(f'service {k8s_object_name(service)} conflict')


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

    if k8s_object_name(service).endswith(training_suffix()):
        return None

    logger.info(f'duplicating {kind(service)}: {k8s_object_name(service)}')

    old_name = k8s_object_name(service)
    service.metadata.name += '-' + training_suffix()
    service.metadata.labels.update({config.PROXY_TAG: 'true'})
    service.spec.selector.update({config.PROXY_TAG: 'true'})

    if check_if_is_node_port(service):
        for port in service.spec.ports:
                update_node_port(port, set_as_none=True)

    try:
        config.coreApi().create_namespaced_service(namespace=config.NAMESPACE, body=duplicate_service(service))
        duplicated_svc[old_name] = service.metadata.name
    except ApiException as e:
        if 409 == e.status:
            logger.warning(f'deployment {service.metadata.name} is already deployed')


def last_applied_svc_configuration(service: V1Service):
    return json.loads(service.metadata.annotations.get('kubectl.kubernetes.io/last-applied-configuratio', '{}'))


def duplicate_service(service: V1Service):
    service.spec.cluster_ip = None
    service.metadata.resource_version = None

    return service


def update_deployment(event):
    if evaluate_event(event):
        last_applied_deployment = inject_proxy_to_deployment(event)
        # duplicate_deployment_for_training(last_applied_deployment)

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
        logger.info(
            f'proxy is present at {kind(deployment)}: {k8s_object_name(deployment)} -- version {resource_version(deployment)}')
        return deployment

    containers = deployment.spec.template.spec.containers
    first_container: V1Container = containers[0]
    res_limits = first_container.resources.limits
    first_port: V1ContainerPort = first_container.ports[0]
    service_port = first_port.container_port

    if res_limits:
        containers.append(proxy_container(
            proxy_port=config.PROXY_PORT,
            metrics_port=config.METRICS_PORT,
            service_port=service_port,
            cpu=res_limits.get('cpu'),
            memory_mb=res_limits.get('memory')
        ))
    else:
        containers.append(proxy_container(
            proxy_port=config.PROXY_PORT,
            metrics_port=config.METRICS_PORT,
            service_port=service_port,
        ))

    patch_body = {
        "kind": deployment.kind,
        "apiVersion": deployment.api_version,
        "metadata": {
            "labels": {"has_proxy": "true", config.PROXY_TAG: "false"},
            # "annotations": {"configmap.reloader.stakater.com/reload": ','.join(
            #     extract_configs_names(deployment).union({config.PROXY_CONFIG_MAP}))}
            "annotations": {
                "reloader.stakater.com/auto": "true"
            }
        },
        "spec": {
            "template": {
                "metadata": {
                    "labels": {config.PROXY_TAG: "false"}
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

    return config.appsApi().patch_namespaced_deployment(deployment.metadata.name, deployment.metadata.namespace,
                                                        patch_body)


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


def container_dict_to_model(c: dict) -> V1Container:
    env_var = lambda name, path: V1EnvVar(name=name,
                                          value_from=V1EnvVarSource(field_ref=V1ObjectFieldSelector(field_path=path)))

    return V1Container(name=c['name'], image=c['image'], image_pull_policy=c['imagePullPolicy'],
                       ports=[V1ContainerPort(container_port=p['containerPort']) for p in c['ports']],
                       env=[
                           V1EnvVar(name='PROXY_PORT', value=c['env'][0]),
                           V1EnvVar(name='METRICS_PORT', value=c['env'][1]),
                           V1EnvVar(name='SERVICE_PORT', value=c['env'][2]),
                           env_var('NODE_NAME', 'spec.nodeName'),
                           env_var('POD_NAME', 'metadata.name'),
                           env_var('POD_NAMESPACE', 'metadata.namespace'),
                           env_var('POD_IP', 'status.podIP'),
                           env_var('POD_SERVICE_ACCOUNT', 'spec.serviceAccountName'),
                       ],
                       env_from=[V1EnvFromSource(config_map_ref=V1ConfigMapEnvSource(name=config.PROXY_CONFIG_MAP))]
                       )


def proxy_container(proxy_port: int, metrics_port: int, service_port: int, cpu:int=None, memory_mb:int=None):
    env_var = lambda name, path: {
        'name': name,
        'valueFrom': {
            'fieldRef': {
                'fieldPath': path
            }
        }
    }

    resources = {}
    if cpu or memory_mb:
        resources = {
            'resources': {
                'limits': {}
            }
        }
        if cpu:
            resources['resources']['limits'].update({'cpu': cpu})
        if memory_mb:
            resources['resources']['limits'].update({'memory': memory_mb})

    proxy = {
        'env': [
            {'name': 'PROXY_PORT', 'value': f'{proxy_port}'},
            {'name': 'METRICS_PORT', 'value': f'{metrics_port}'},
            {'name': 'SERVICE_PORT', 'value': f'{service_port}'},
            env_var('NODE_NAME', 'spec.nodeName'),
            env_var('POD_NAME', 'metadata.name'),
            env_var('POD_NAMESPACE', 'metadata.namespace'),
            env_var('POD_IP', 'status.podIP'),
            env_var('POD_SERVICE_ACCOUNT', 'spec.serviceAccountName'),
        ],
        'envFrom': [{
            'configMapRef': {
                'name': config.PROXY_CONFIG_MAP
            }
        }],
        'image': config.PROXY_IMAGE,
        'imagePullPolicy': 'IfNotPresent',
        'name': config.PROXY_NAME,
        'ports': [{'containerPort': proxy_port}, {'containerPort': metrics_port}],

    }
    if resources:
        proxy.update(resources)

    return proxy


def duplicate_deployment_for_training(deployment: V1Deployment) -> str:
    if deployment.metadata.name.endswith(training_suffix()):
        return None

    old_name = deployment.metadata.name
    deployment.spec.replicas = 1
    deployment.metadata.name += training_suffix()

    deployment.metadata.labels.update({config.PROXY_TAG: 'true'})
    deployment.spec.selector.match_labels.update({config.PROXY_TAG: 'true'})
    deployment.spec.template.metadata.labels.update({config.PROXY_TAG: 'true'})

    # deployment.spec.selector.match_labels.update({config.PROXY_TAG: 'false'})
    # deployment.metadata.labels.update({config.PROXY_TAG: 'false'})
    # deployment.spec.template.metadata.labels.update({config.PROXY_TAG: 'false'})

    configs = extract_configs_names(deployment)
    duplicating_deployment_configs(configs)

    config_maps = append_suffix_to_configs_names(deployment, suffix=training_suffix())
    deployment.metadata.annotations.update(
        {"configmap.reloader.stakater.com/reload": ','.join(config_maps.union({config.PROXY_CONFIG_MAP}))})

    duplicate_deployment(deployment)

    try:
        config.appsApi().create_namespaced_deployment(namespace=config.NAMESPACE,
                                                        body=duplicate_deployment(deployment))
        duplicated_dep[old_name] = deployment.metadata.name
    except ApiException as e:
        if 409 == e.status:
            logger.warning(f'deployment {deployment.metadata.name} is already deployed')
        else:
            logger.warning(f'>>> {deployment}')

    return deployment.metadata.name

def extract_configs_names(deployment: V1Deployment) -> set:
    spec: V1PodSpec = deployment.spec.template.spec
    containers = spec.containers

    config_maps_to_return = set()
    container: V1Container
    for container in containers:
        if isinstance(container, dict):
            container = container_dict_to_model(container)

        if config.PROXY_NAME == container.name:
            continue

        if container.env:
            for var in container.env:
                value_from = var.value_from
                if value_from:
                    config_map_key_ref = value_from.config_map_key_ref
                    if config_map_key_ref:
                        config_maps_to_return.add(config_map_key_ref.name)

        if container.env_from:
            for var in container.env_from:
                config_map_ref = var.config_map_ref
                if config_map_ref:
                    config_maps_to_return.add(config_map_ref.name)

    if spec.volumes:
        for volume in spec.volumes:
            config_map = volume.config_map
            if config_map:
                config_maps_to_return.add(config_map.name)

    return config_maps_to_return


def append_suffix_to_configs_names(deployment: V1Deployment, suffix=training_suffix()):
    spec: V1PodSpec = deployment.spec.template.spec
    containers = spec.containers

    config_maps_to_return = set()
    container: V1Container
    for container in containers:
        if isinstance(container, dict):
            container = container_dict_to_model(container)

        if config.PROXY_NAME == container.name:
            continue

        if container.env:
            for var in container.env:
                value_from = var.value_from
                if value_from:
                    config_map_key_ref = value_from.config_map_key_ref
                    if config_map_key_ref:
                        config_map_key_ref.name += suffix
                        config_maps_to_return.add(config_map_key_ref.name)

        if container.env_from:
            for var in container.env_from:
                config_map_ref = var.config_map_ref
                if config_map_ref:
                    config_map_ref.name += suffix
                    config_maps_to_return.add(config_map_ref.name)

    if spec.volumes:
        for volume in spec.volumes:
            config_map = volume.config_map
            if config_map:
                config_map.name += suffix
                config_maps_to_return.add(config_map.name)

    return config_maps_to_return


def duplicating_deployment_configs(config_maps):
    config_map: V1ConfigMap
    for config_map in config.coreApi().list_namespaced_config_map(namespace=config.NAMESPACE).items:
        logger.debug(f'duplicating cm: {config_map.metadata.name} data: {config_map.data}')
        if config_map.metadata.name in config_maps and not config_map.metadata.name.endswith(training_suffix()):
            old_name = config_map.metadata.name
            config_map.metadata.name += training_suffix()
            try:
                config.coreApi().create_namespaced_config_map(namespace=config.NAMESPACE,
                                                                body=duplicate_config_map(config_map))
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
    w = watch.Watch()
    for event in w.stream(list_to_watch, config.NAMESPACE):
        logger.info("Event: %s %s %s %s" % (
            event_type(event), kind(event['object']), k8s_object_name(event['object']), resource_version(event['object'])))
        try:
            handler(event)
        except Exception:
            logger.exception('error at event loop')
            w.stop()


def event_type(event):
    return event['type']


def namespace(k8s_object):
    return k8s_object.metadata.namespace


def delete_services(event: V1Event):
    delete_object(event, duplicated_svc, client.CoreV1Api().delete_namespaced_service)


def delete_configs_maps(event: V1Event):
    delete_object(event, duplicated_cm, client.CoreV1Api().delete_namespaced_config_map)


def delete_deployments(event: V1Event):
    delete_object(event, duplicated_dep, client.AppsV1Api().delete_namespaced_deployment)


def delete_object(event: V1Event, duplicates: dict, listing_fn):
    k8s_object = event['object']
    if 'DELETED' == event_type(event):
        k8s_obj_name = k8s_object_name(k8s_object)
        duplicated_object_key = k8s_obj_name
        if k8s_obj_name.endswith(config.PROXY_TAG):
            duplicated_object_key = k8s_obj_name.replace(config.PROXY_TAG, '')

        k8s_obj_name = duplicates.get(duplicated_object_key, '')
        logger.debug(f'event: {event["object"].metadata.name}')
        logger.debug(f'going to delete [{k8s_obj_name}] at {duplicates}')
        if k8s_obj_name:
            logger.info(f'deleting {kind(k8s_object)}: {k8s_obj_name}')
            try:
                listing_fn(name=k8s_obj_name, namespace=namespace(k8s_object))
            except ApiException as e:
                if 404 == e.status:
                    logger.warning(f'object: [{k8s_obj_name}] was already deleted')
            if duplicated_object_key in duplicates:
                del duplicates[duplicated_object_key]


def init(loop:EventLoop, namespace:str=config.NAMESPACE):
    # initializing
    loop.register('services-injector', ListToWatch(func=client.CoreV1Api().list_namespaced_service, namespace=namespace), update_service)

    loop.register('deployment-injector', ListToWatch(func=client.AppsV1Api().list_namespaced_deployment, namespace=namespace), update_deployment)

    # garbage collection
    loop.register('services-gc', ListToWatch(func=client.CoreV1Api().list_namespaced_service, namespace=namespace), delete_services)

    loop.register('configmaps-gc', ListToWatch(func=client.CoreV1Api().list_namespaced_config_map, namespace=namespace), delete_configs_maps)

    loop.register('deployments-gc', ListToWatch(func=client.AppsV1Api().list_namespaced_deployment, namespace=namespace), delete_deployments)

    logger.debug('waiting 1s as safeguard to enusre all loops are going to be registered properly')
    time.sleep(1)


if __name__ == '__main__':
    try:
        loop = EventLoop(config.executor())
        init(loop)
    except Exception:
        logger.exception('error injector loop')
    # finally:
    #     loop.shutdown()
    #     config.shutdown()

