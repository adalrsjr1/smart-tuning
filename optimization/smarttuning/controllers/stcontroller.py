import kubernetes as k8s
from kubernetes.client.models import *
from pprint import pprint
import config
import json
from controllers.k8seventloop import EventLoop, ListToWatch
from controllers.searchspacemodel import SearchSpaceModel

def init(loop: EventLoop):
    loop.register('searchspace-controller',
                  ListToWatch(func=k8s.client.CustomObjectsApi().list_namespaced_custom_object,
                              namespace=config.NAMESPACE,
                              group='smarttuning.ibm.com',
                              version='v1alpha2',
                              plural='searchspaces'), callback)

def callback(event: dict):
    if event['type'] == 'ADD':
        pass
    elif event['type'] == 'DELETE':
        pass
    SearchSpaceModel(event['object'])


