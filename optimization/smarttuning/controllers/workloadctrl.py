import logging
import threading
import time
import math

from kubernetes.client import ApiException, V2beta2HorizontalPodAutoscaler, V2beta2HorizontalPodAutoscalerSpec, \
    V2beta2CrossVersionObjectReference
from kubernetes.watch import Watch

import config
from controllers.k8seventloop import EventLoop, ListToWatch
from models.workload import Workload
from sampler import PrometheusSampler

logger = logging.getLogger(config.WORKLOAD_LOGGER)
logger.setLevel(logging.INFO)

workloads: set[Workload] = set()


def init(loop: EventLoop):
    e = threading.Event()
    hpa = config.hpaApi()
    name = config.HPA_NAME
    if not name:
        return

    def initialize():
        # workaround to make hpa watcher work properly
        # https://github.com/kubernetes-client/python/issues/1098
        while True:
            try:
                hpa.read_namespaced_horizontal_pod_autoscaler(name, 'default')
                break
            except Exception:
                logger.info(f'waiting for HPA: {name}')
                time.sleep(2)
        logger.info(f'HPA: {name} deployed')
        e.set()

    t = threading.Thread(target=initialize, name='workloadctrl-workaround')
    t.start()

    e.wait()
    namespace = config.NAMESPACE
    loop.register('workload-controller',
                  ListToWatch(func=hpa.list_namespaced_horizontal_pod_autoscaler,
                              namespace=namespace), workload_controller_wrapper(name))


__lock = threading.Event()
__rlock = threading.RLock()
__app_name = ''


def app_name() -> str:
    return __app_name


def wait():
    logger.debug('waiting for HPA be ready')
    __lock.wait()


def release():
    with __rlock:
        __lock.set()


def workload() -> Workload:
    with __rlock:
        if 'RPS' == config.WORKLOAD_CLASSIFIER:
            return new_rps_based_workload()
        elif 'HPA' == config.WORKLOAD_CLASSIFIER:
            return new_replica_based_workload()
        else:
            return Workload('')


def workload_controller_wrapper(name):
    def workload_controller(event):

        # print(event['type'], event['object'].metadata.name)
        first = False
        if 'ADDED' == event['type']:
            # print(event['type'], event['object'].metadata.name)
            pass
        elif 'DELETED' == event['type']:
            pass
        elif 'MODIFIED' == event['type']:
            hpa_obj: V2beta2HorizontalPodAutoscaler = event['object']
            spec: V2beta2HorizontalPodAutoscalerSpec = hpa_obj.spec
            target: V2beta2CrossVersionObjectReference = spec.scale_target_ref
            global __app_name
            if not __app_name:
                __app_name = target.name
            print(name, event['type'], hpa_obj.metadata.name,
                  [condition.reason for condition in event['object'].status.conditions])
            if name == event['object'].metadata.name:
                status = event['object'].status
                conditions = status.conditions
                for condition in conditions:
                    if 'ReadyForNewScale' == condition.reason:
                        logger.debug(
                            f'HPA Event on {__app_name}: {event["type"]} {condition.reason} {condition.type}={condition.status} '
                            f'{status.current_replicas}/{status.desired_replicas}')
                release()

    return workload_controller


def new_replica_based_workload() -> Workload:
    workload = None
    try:
        client = config.hpaApi()
        hpa = client.read_namespaced_horizontal_pod_autoscaler(name=config.HPA_NAME,
                                                               namespace=config.NAMESPACE)
        status = hpa.status
        # add +1 due to training replica
        n_replicas = status.current_replicas

        workload = Workload(f'workload_{n_replicas}', data=n_replicas)
    except ApiException:
        logger.exception('cannot sample HPA info')
    finally:
        logger.debug(f'sampling workload: {workload}')
        return workload


def new_rps_based_workload() -> Workload:
    workload = None

    def classify_truput(truput: float, bands: None):
        if not bands:
            bands = []
        # if truput % band_width < deviation:
        #     classification = str(int(truput // band_width))
        # else:
        #     classification = str(int((truput // band_width) + 1))
        # return classification

        # bands = [200, 600, 1500]

        min_d = float('inf')
        min_b = 0
        for idx, band in enumerate(bands):
            d = math.sqrt((float(band) - truput) ** 2)
            if d < min_d:
                min_b = idx
                min_d = d

        return str(min_b)

    try:
        # classification based on both training and production truput
        ps_prod = PrometheusSampler(app_name(), config.WAITING_TIME * config.SAMPLE_SIZE, aggregation_function='sum')
        # ps_train = PrometheusSampler(app_name()+config.PROXY_TAG, config.WAITING_TIME * config.SAMPLE_SIZE, aggregation_function='sum')
        # truput = ps_prod.metric().throughput() + ps_train.metric().throughput()
        truput = ps_prod.metric().throughput()
        workload = Workload(f'workload_{classify_truput(truput, config.WORKLOAD_BANDS)}',
                            data=truput)
    except Exception:
        logger.exception('cannot sample rps')
    finally:
        logger.debug(f'sampling workload: {workload}')
        return workload


if __name__ == '__main__':
    print('wctlr')
    hpa = config.hpaApi()
    w = Watch()

    # while True:
    #     try:
    #         hpa.read_namespaced_horizontal_pod_autoscaler('daytrader-service', 'default')
    #         break
    #     except:
    #         print('waiting...')
    #         time.sleep(2)
    #
    # try:
    #     print('ok')
    #     stream = w.stream(hpa.list_namespaced_horizontal_pod_autoscaler, 'default')
    #     for event in stream:
    #         print("[%s] Event: %s" % (event['type'], event['object'].metadata.name))
    # except:
    #     traceback.print_exc()

    # try:
    loop = EventLoop(config.executor())
    init(loop)
    # except Exception:
    #     logger.exception('error injector loop')
    # finally:
    #     loop.shutdown()
    #     config.shutdown()