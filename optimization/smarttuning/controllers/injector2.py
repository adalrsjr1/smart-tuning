import copy
import time

from kubernetes import client
from kubernetes.client import AppsV1Api, ApiException, CoreV1Api
from kubernetes.client.models import *
import logging

from kubernetes.watch import watch

import config
from controllers.k8seventloop import EventLoop, ListToWatch

logger = logging.getLogger(config.INJECTOR_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)

duplicated_dep = {}

def training_suffix():
    return config.PROXY_TAG


def handle_event(event, apps, core):
    obj = event['object']
    if isinstance(obj, V1Deployment):
        handle_deployment(event)
    elif isinstance(obj, V1Service):
        handle_service(obj, event['type'], core)
    else:
        logger.warning(
            f'there is not handler for event: {event["type"]} {obj.kind} {obj.metadata.name} {obj.metadata.namespace}')


def handle_deployment(event):
    event_type = event['type']
    dep = event['object']
    apps = config.appsApi()
    core = config.coreApi()
    if dep.metadata.annotations.get('injection.smarttuning.ibm.com', 'false').upper() != 'TRUE':
        return

    if dep.metadata.name.endswith(training_suffix()):
        return

    def extract_original_cm_names_and_update_training_cm_names(_dep: V1Deployment):
        spec: V1PodSpec = _dep.spec.template.spec
        containers = spec.containers

        container: V1Container
        _config_maps = set()

        for container in containers:
            if config.PROXY_NAME == container.name:
                continue

            if container.env:
                for var in container.env:
                    value_from = var.value_from
                    if value_from:
                        config_map_key_ref = value_from.config_map_key_ref
                        _config_maps.add(config_map_key_ref.name)
                        config_map_key_ref.name += training_suffix()

            if container.env_from:
                for var in container.env_from:
                    config_map_ref = var.config_map_ref
                    if config_map_ref and not config_map_ref.name.endswith(training_suffix()):
                        _config_maps.add(config_map_ref.name)
                        config_map_ref.name += training_suffix()

        if spec.volumes:
            for volume in spec.volumes:
                config_map = volume.config_map
                if config_map and not config_map.name.endswith(training_suffix()):
                    _config_maps.add(config_map.name)
                    config_map.name += training_suffix()

        return _dep, _config_maps

    train_dep = copy.deepcopy(dep)
    train_dep.metadata.annotations.update({"reloader.stakater.com/auto": "true"})
    train_dep, config_maps = extract_original_cm_names_and_update_training_cm_names(train_dep)
    train_dep.metadata = V1ObjectMeta(
        annotations=train_dep.metadata.annotations,
        labels=train_dep.metadata.labels,
        name=train_dep.metadata.name + training_suffix(),
        namespace=train_dep.metadata.namespace)
    train_dep.spec.replicas = 1

    if 'ADDED' == event_type:
        def add_dep(dep: V1Deployment, cm_names: set[str]):
            for cm_name in cm_names:
                cm = core.read_namespaced_config_map(cm_name, dep.metadata.namespace)
                cm.metadata = V1ObjectMeta(name=cm_name + training_suffix(),
                                           namespace=dep.metadata.namespace)
                try:
                    core.create_namespaced_config_map(namespace=dep.metadata.namespace, body=cm)
                except ApiException as e:
                    if 409 == e.status:
                        logger.exception(f'cm {cm.metadata.name}.{cm.metadata.namespace} already exists')

            result = None
            try:
                if config.TWO_SERVICES:
                    dep.spec.selector.match_labels.update({config.PROXY_TAG: 'true'})
                    dep.spec.template.metadata.labels.update({config.PROXY_TAG: 'true'})
                result = apps.create_namespaced_deployment(dep.metadata.namespace, body=dep)
            except ApiException as e:
                if 409 == e.status:
                    logger.warning('training replicas is already running')
                else:
                    logger.exception('')
            return result

        result = add_dep(train_dep, config_maps)
        if result:
            duplicated_dep[dep.metadata.name] = train_dep.metadata.name
        return result

    elif 'MODIFIED' == event_type:
        if "reloader.stakater.com/auto" not in dep.metadata.annotations:
            dep.metadata.annotations.update({"reloader.stakater.com/auto": "true"})
            dep.metadata = V1ObjectMeta(
                annotations=dep.metadata.annotations,
                labels=dep.metadata.labels,
                name=dep.metadata.name,
                namespace=dep.metadata.namespace)

            apps.patch_namespaced_deployment(
                name=dep.metadata.name,
                namespace=dep.metadata.namespace,
                body=dep)

    elif 'DELETED' == event_type:
        def del_dep(dep: V1Deployment, cm_names: set[str]):
            for cm_name in cm_names:
                core.delete_namespaced_config_map(cm_name + training_suffix(), dep.metadata.namespace)

            result = None
            try:
                result = apps.delete_namespaced_deployment(train_dep.metadata.name, train_dep.metadata.namespace)
            except ApiException as e:
                if 404 == e.status:
                    logger.warning('there is no training replica to be removed')

            return result

        result = del_dep(train_dep, config_maps)
        if result:
            del duplicated_dep[dep.metadata.name]
        return result


def handle_service(event):
    event_type = event['type']
    svc = event['object']
    core = config.coreApi()
    if not config.TWO_SERVICES:
        return

    if svc.metadata.annotations.get('injection.smarttuning.ibm.com', 'false').upper() != 'TRUE':
        return

    if svc.metadata.name.endswith(training_suffix()):
        return

    svc.metadata.labels.update({config.PROXY_TAG: 'true'})
    svc.metadata = V1ObjectMeta(
        name = svc.metadata.name + '-' + training_suffix(),
        namespace = svc.metadata.namespace,
        labels = svc.metadata.labels,
        annotations = svc.metadata.annotations,
    )

    if 'ADDED' == event_type:
        svc.spec.selector.update({config.PROXY_TAG: 'true'})
        svc.spec.cluster_ip = None
        if 'NodePort' == svc.spec.type:
            port: V1ServicePort
            for port in svc.spec.ports:
                port.node_port = None
        try:
            return core.create_namespaced_service(namespace=svc.metadata.namespace, body=svc)
        except ApiException as e:
            if 409 == e.status:
                logger.warning(f'deployment {svc.metadata.name} is already deployed')
            else:
                logger.exception('')
    elif 'DELETED' == event_type:
        try:
            return core.delete_namespaced_service(name=svc.metadata.name, namespace=svc.metadata.namespace)
        except ApiException as e:
            logger.exception('')

def init(loop: EventLoop, namespace: str = config.NAMESPACE):
    # initializing
    loop.register('service-injector',
                  ListToWatch(func=config.coreApi(), namespace=namespace), handle_service)
    loop.register('deployment-injector',
                  ListToWatch(func=config.coreApi(), namespace=namespace), handle_deployment)

    time.sleep(1)

if __name__ == '__main__':
    config.init_k8s()
    apps = config.appsApi()
    core = config.coreApi()
    w = watch.Watch()
    # for event in w.stream(apps.list_namespaced_deployment, 'quarkus'):
    for event in w.stream(core.list_namespaced_service, 'quarkus'):
        handle_event(event, apps, core)
