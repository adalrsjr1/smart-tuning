from kubernetes import client
from kubernetes.client.models import *


class FakeObject:
    def __init__(self, filename):
        with open(filename) as file:
            self.data = file.read()


class FakeCustomObject(FakeObject):
    def __init__(self, filename):
        super(FakeCustomObject, self).__init__(filename)


class FakeDeployment(FakeObject):
    def __init__(self, filename):
        super(FakeDeployment, self).__init__(filename)


class FakeConfigMap(FakeObject):
    def __init__(self, filename):
        super(FakeConfigMap, self).__init__(filename)


def mock_acmeair_search_space() -> dict:
    c = client.CustomObjectsApi()
    return c.api_client.deserialize(FakeCustomObject('fake-acmeair-ss.json'),
                                    'object')


def mock_acmeair_search_space_dep_continous() -> dict:
    c = client.CustomObjectsApi()
    return c.api_client.deserialize(FakeCustomObject('fake-acmeair-ss_dep-continuous.json'),
                                    'object')


def mock_acmeair_deployment(*kwargs) -> V1Deployment:
    c = client.AppsV1Api()
    return c.api_client.deserialize(FakeDeployment('fake-acmeair-dep.json'),
                                    'V1Deployment')


def mock_acmeair_app_cm() -> V1ConfigMap:
    c = client.CoreV1Api()
    return c.api_client.deserialize(FakeConfigMap('fake-acmeair-config-app-cm.json'),
                                    'V1ConfigMap')


def mock_acmeair_jvm_cm() -> V1ConfigMap:
    c = client.CoreV1Api()
    return c.api_client.deserialize(FakeConfigMap('fake-acmeair-config-jvm-cm.json'),
                                    'V1ConfigMap')


def mock_daytrader_ss() -> FakeCustomObject:
    c = client.CustomObjectsApi()
    return c.api_client.deserialize(FakeCustomObject('daytrader-ss.json'),
                                    'object')
