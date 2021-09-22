# Walkthrough

This guide will details how to deploy and set up an environment to tune
application using Smart Tuning.

## Infra needed

#### Local K8s

It is expected that SmartTuning runs on any Kubernetes deployment +v1.18.
Although, we carried out all experiments either on:

* [Kind](https://kind.sigs.k8s.io/)
* [Azure](https://azure.microsoft.com/en-us/services/kubernetes-service/)

#### Prometheus

Any instance of Prometheus should work for SmartTuning, since it is configured
to scrap both Kubernetes API and Kubernetes metrics server specified in these
[manifests](./optimization/prometheus). Furthermore, if you need to use the
custom SmartTuning proxy for observability, please refer to `- job-name:
'smarttuning-services'` and `- job-name: 'smarttuning-metrics'` in [this
configuration](./optimization/prometheus/2-monitoring-configmap.yml)

#### Metrics server

SmartTuning relies on Metrics server to sample some specific related to resource
consumption. Use the manifests in
[kube-state-metrics](./optimization/kube-state-metrics) and
[metrics-server](./optimization/metrics-server) for this.

#### Grafana

Deploy Grafana +7.5.8 if using any of the dashboards available in [this
repository](./optimization/grafana).

#### MongoDB

SmartTuning requires Mongo:4.4 to work. For keep tracking of tuning, uses
[Mongo Express](https://github.com/mongo-express/mongo-express) to visualize the
data for every iteration.

#### Reloader

SmartTuning deeply relies on [Reloader](https://github.com/stakater/Reloader) to
automatically update the configurations in the application running. Please,
refers to [this](https://github.com/stakater/Reloader#vanilla-manifests)
installation guide.

#### Search Space CRD

For tuning an application SmartTuning needs to know what are the parameters and
their boundaries. This specification is made through this (CRD)[optimization/manifests/search-space/search-space-crd-2.yaml]

### Summary

To quickly set up the infrastructure needed for SmartTuning just run
[./optimization/manifests/monitoring-deploy.sh](./optimization/manifests/monitoring-deploy.sh).

```
./monitoring-deploy.sh $HOME/.kube/config
```

Use a different `kubeconfig` path if needed.

## Application

SmartTuning has been experimented with the following three applications,
although it is expected that it works with any application that exposes its
configurations through environment variables.

* [AcmeAir](https://github.com/adalrsjr1/acmeair-monolithic-java)
* [Daytrader](https://github.com/adalrsjr1/sample.daytrader8)
* [Quarkus Demo](https://github.com/adalrsjr1/quarkusRestCrudDemo)

In this document we will walkthrough the deployment of Quarkus Demo, the
simplest application to try. The complete deployment is [here](./optimization/manifests/icse/quarkus).

### Deployment

All applications to be tuned by SmartTuning need to be annotated as
`injection.smarttuning.ibm.com: "true"` so SmartTuning can automatically
configure and create a training replica automatically.

#### Quarkus Demo

```
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: quarkus-service
  annotations:
    injection.smarttuning.ibm.com: "true"
  name: quarkus-service
  namespace: quarkus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: quarkus-service
      smarttuning: "false"
  strategy: {}
  template:
    metadata:
      labels:
        app: quarkus-service
        smarttuning: "false"
      annotations:
        prometheus.io/scrape: 'true'
        prometheus.io/port: '9080'
    spec:
      containers:
      - image: smarttuning/rest-crud-quarkus-native
        imagePullPolicy: IfNotPresent
        name: rest-crud-quarkus-native
        ports:
          - containerPort: 9080
        env:
          - name: "quarkus.http.port"
            value: "9080"
          - name: "quarkus.datasource.jdbc.url"
            value: "jdbc:postgresql://postgres-quarkus-rest-http-crud-svc.quarkus.svc.cluster.local/rest-crud"
        resources:
          limits:
            cpu: 1
            memory: "1024Mi"
        envFrom:
          - configMapRef:
              name: quarkus-cm-app
          - configMapRef:
              name: quarkus-cm-jvm
```

#### Database

Heads up with the number of available connections open in the Database `-N` and
the max number of connections that the application can open to the database.
Besides, take care with the max number of replicas and the max number of
connection opened in the application: `N >= n * app_conn`.

```
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: postgres-quarkus-rest-http-crud
  name: postgres-quarkus-rest-http-crud
  namespace: quarkus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-quarkus-rest-http-crud
  template:
    metadata:
      labels:
        app: postgres-quarkus-rest-http-crud
    spec:
      containers:
      - image: postgres:10.5
        name: postgres
        args: ["-N", "1000", "-B", "8192MB"]
        imagePullPolicy: IfNotPresent
        resources: {}
        env:
          - name: POSTGRES_USER
            value: "restcrud"
          - name: POSTGRES_PASSWORD
            value: "restcrud"
          - name: POSTGRES_DB
            value: "rest-crud"
        ports:
          - containerPort: 5432
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: postgres-quarkus-rest-http-crud-svc
  name: postgres-quarkus-rest-http-crud-svc
  namespace: quarkus
spec:
  ports:
  - name: "pg"
    port: 5432
    protocol: TCP
    targetPort: 5432
  selector:
    app: postgres-quarkus-rest-http-crud
  type: NodePort
```

### Service

The service also need to be annotated with `injection.smarttuning.ibm.com:
"true"` as the application.

```
---
apiVersion: v1
kind: Service
metadata:
  name: quarkus-svc
  namespace: quarkus
  labels:
    app: quarkus-svc
  annotations:
    injection.smarttuning.ibm.com: "true"
spec:
  ports:
  - port: 9080
    targetPort: 9080
    name: http
  selector:
    app: quarkus-service
    smarttuning: "false"
  type: NodePort
---
```

### ConfigMaps

Note that the application deployment refers to different `ConfigMap`s. These
manifests holds all configurations that will be tuned by SmartTuning during the
optimization process.

The only special attention when setting the ConfigMaps is that all parameters in
the `search-space` also must exists in the ConfigMaps, otherwise, SmartTuning
can't sample different values for them.

Despite not being necessary, it is recommended to have one ConfigMap for
application layer, e.g., one for JVM, another for Quarkus, and so on.

#### Quarkus

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: quarkus-cm-app
  namespace: quarkus
data:
  quarkus.datasource.jdbc.min-size: "0"
  quarkus.datasource.jdbc.max-size: "20"
  quarkus.datasource.jdbc.max-lifetime: "1800"
  quarkus.datasource.jdbc.idle-removal-interval: "300"
  quarkus.vertx.event-loops-pool-size: "2"
  quarkus.vertx.worker-pool-size: "20"
  quarkus.http.io-threads: "4"
  quarkus.http.idle-timeout: "1800"
  quarkus.http.so-reuse-port: "false"

```

#### JVM

If JVM is configured using a `jvm.options` file, like in AcmeAir and Daytrader,
not all parameter can be tuned for technical limitations in the current
SmartTuning implementation. For more details about this limitation, refers to
`dict_to_jvmoptions` and `jvmoptions_to_dict` in
[here](./optimization/smarttuning/controllers/searchspacemodel.py). This is the
list of available parameters to tune:

```
    -XX:+UseContainerSupport
    -Xgcpolicy:gencon
    -Xtune:virtualized
    -Xms8m
    -Xmx256m
    -Xmns2m
    -Xmnx64m
    -XX:SharedCacheHardLimit=32m
    -Xscmx=16m
```

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: quarkus-cm-jvm
  namespace: quarkus
data:
  XMX: "128"
  XMN: "110"
  XMS: "100"
```

### SearchSpace

The search space is the manifest necessary for defining the boundaries of all
parameters that SmartTuning aims to tune, refers to
[search-space.yaml](./optimization/manifests/icse/quarkus/search-space/quarkus-ss.yaml)
for a complete specification. When SmartTuning identifies that a search-space is
  deployed, it automatically start the process to create a training replica of
  the application (the search-space manifest deployment is the trigger).

The search-space has two main parts, `spec` and `data`. The spec, see below,
maps which are the `configMaps` and the application `deployment` that
SmartTuning can touch during the tuning. 

```
spec:
  deployment: quarkus-service
  service: quarkus-svc
  namespace: "quarkus"
  manifests:
    - name: quarkus-service
      type: "deployment"
    - name: quarkus-cm-app
      type: "configMap"
    - name: quarkus-cm-jvm
      type: "configMap"
```

The `data` part, specifies what parameters are going to be tuned, their types,
dependencies, and up and lower boundaries. There are two classes of parameters
(tunables) available to specify boundaries: `option` and `number`. `option` in a
discrete and finite list of values of a same type (usually strings). This class
of tunable does not allow specify any dependency. During a tuning, SmartTuning
will sample one of these values accordingly. Conversely, `number` defines a
range of values that SmartTuning will sample during the tuning. Also, `number`
accepts the specification of dependencies, i.e., the upper or lower boundaries
of a parameter can depends of the values sampled for others. This dependency
specification can be just the name of another tunable, or an arithmetic formula
expressed as a [reverse polish
notation](https://en.wikipedia.org/wiki/Reverse_Polish_notation) --- using this
notation for the sake of simplicity for interpreter implementation. It is always
necessary to specify the upper and lower values, even if using any dependency
for them, see below.

```
...
  - name: quarkus-cm-jvm
    tunables:
      number:
        - name: "XMS"
          lower:
            value: 8
          upper:
            value: 896
            dependsOn: "memory 0.4 *"
...
```


### HPA

When all manifests has been deployed, SmartTuning will waits for HPA start to
sample metrics from the metrics server to start tuning the application. HPA is
the last trigger to initialize SmartTuning initalization. It is fundamental that
the name of HPA is the name of the application deployment.

```
---
kind: HorizontalPodAutoscaler
apiVersion: autoscaling/v2beta2
metadata:
  name: quarkus-service
  namespace: quarkus
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quarkus-service
  minReplicas: 1
  maxReplicas: 8
  metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 49

```

## SmartTuning

For SmartTuning be able to acces the Kubernetes API and perform changes on
the app's configMaps, it is necessary to give it autorization through RABC.

```
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: tuning
  namespace: quarkus
rules:
  - apiGroups: [ "*" ]
    resources: [ "*" ]
    verbs: [ "*" ]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: tuning
  namespace: quarkus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: tuning
subjects:
  - kind: ServiceAccount
    name: default
    namespace: quarkus
---

```

Besides, SmartTuning the specification of Prometheus queries for observe the
application, see below. The queries and their names, and the objective and
penalization functions are application dependent. For more details about how to
define these queries, refer to [deployment
documentation](./optimization/manifests/README.md)

##### Prometheus queries

```
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: smarttuning-sampler-config
  namespace: quarkus
data:
  sampler.json: |
    {
      "objective": "-(penalization)*(throughput/((memory_limit/(2**10) + cpu_limit))) * 1/(1+process_time)",
      "penalization": "1",
      "metrics": [
        {
          "name": "cpu",
          "query": "f'sum(rate(container_cpu_usage_seconds_total{{id=~\".*kubepods.*\",pod=~\"{podname}-.*\",name!~\".*POD.*\",namespace=\"{namespace}\",container=\"\"}}[{interval}s]))'",
          "datasource": "prom"
        },
        {
          "name": "cpu_limit",
          "query": "f'sum(sum_over_time(container_spec_cpu_quota{{id=~\".*kubepods.*\",pod=~\"{podname}-.*\",name!~\".*POD.*\",namespace=\"{namespace}\",container=\"\"}}[{interval}s])) / avg(sum_over_time(container_spec_cpu_period{{id=~\".*kubepods.*\",pod=~\"{podname}-.*\",name!~\".*POD.*\",namespace=\"{namespace}\",container=\"\"}}[{interval}s]))'",
          "datasource": "prom"
        },
        {
          "name": "memory",
          "query": "f'sum(max_over_time(container_memory_working_set_bytes{{id=~\".*kubepods.*\",pod=~\"{podname}-.*\",name!~\".*POD.*\",namespace=\"{namespace}\",container=\"\"}}[{interval}s]))/(2^20)'",
          "datasource": "prom"
        },
        {
          "name": "memory_limit",
          "query": "f'sum(max_over_time(container_spec_memory_limit_bytes{{id=~\".*kubepods.*\",pod=~\"{podname}-.*\",name!~\".*POD.*\",namespace=\"{namespace}\",container=\"\"}}[{interval}s]))/(2^20)'",
          "datasource": "prom"
        },
        {
          "name": "throughput",
          "query": "f'sum(rate(base_REST_request_total{{pod=~\"{podname}-.*\",class=\"com.acme.crud.FruitResource\"}}[{interval}s]))'",
          "datasource": "prom"
        },
        {
          "name": "process_time",
          "query": "f'sum(deriv(base_REST_request_elapsedTime_seconds{{pod=~\"{podname}-.*\",class=\"com.acme.crud.FruitResource\"}}[{interval}s]))'",
          "datasource": "prom"
        }
      ]
    }
---
```

###### Smarttuning Configuration

For more details about how to configure SmartTuning, refer to [deployment
documentation](./optimization/manifests/README.md)

```
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: smarttuning-sampling-prom-queries
  namespace: quarkus
data:
  Q_CPU: f'sum(rate(container_cpu_usage_seconds_total{{id=~".*kubepods.*",pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",container=""}}[{self.interval}s]))'
  Q_CPU_L: f'sum(sum_over_time(container_spec_cpu_quota{{id=~".*kubepods.*",pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",container=""}}[{self.interval}s])) / avg(sum_over_time(container_spec_cpu_period{{id=~".*kubepods.*",pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",container=""}}[{self.interval}s]))'
  Q_MEM: f'sum(max_over_time(container_memory_working_set_bytes{{id=~".*kubepods.*",pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",container=""}}[{self.interval}s]))/(2^20)'
  Q_MEM_L: f'sum(max_over_time(container_spec_memory_limit_bytes{{id=~".*kubepods.*",pod=~"{self.podname}-.*",name!~".*POD.*",namespace="{self.namespace}",container=""}}[{self.interval}s]))/(2^20)'
  Q_THRUPUT: f'sum(rate(vendor_servlet_request_total{{pod=~"{self.podname}.*",servlet!~"com_ibm_ws_microprofile.*|.*Trade.*"}}[{self.interval}s]))'
  Q_RESP_TIME: f'avg(deriv(vendor_servlet_responseTime_total_seconds{{pod=~"{self.podname}-.*",servlet!~"com_ibm_ws_microprofile.*"}}[{self.interval}s]))'
  Q_ERRORS: f'0'
  Q_REPLICAS: f'avg_over_time(kube_deployment_spec_replicas{{namespace="{self.namespace}", deployment="{self.podname}"}}[{self.interval}s])'
  Q_CONNS: f'sum(rate(smarttuning_active_conns{{pod=~"{self.podname}-*",name!~".*POD.*", state="new"}}[{self.interval}s]))'
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: smarttuning-config
  namespace: quarkus
data:
  NO_PROXY: "True"
  LOGGING_LEVEL: "NOTSET"
  SAMPLER_CONFIG: "/etc/sampler-config/sampler.json"
  TWO_SERVICES: "False"
  RESTART_IF_CFG_DOESNT_CHANGE: "False"
  WORKLOAD_TIMEOUT: "60"
  WORKLOAD_CLASSIFIER: "CM" #HPA or RPS or CM
  JMETER_CFG_WORKLOAD: "JUSERS"
  JMETER_CM: "jmeter-cm"
  WORKLOAD_BANDS: "0"
  FAIL_FAST: "False"
  MOCK: "False"
  PRINT_CONFIG: "True"
  # mongo config
  MONGO_ADDR: 'mongo-workload-service.default.svc.cluster.local'
  MONGO_PORT: '27017'
  MONGO_DB: 'quarkus'
  # prometheus config
  ST_METRICS_PORT: '9090'
  PROMETHEUS_ADDR: 'prometheus-service.kube-monitoring.svc.cluster.local'
  PROMETHEUS_PORT: '9090'
  SAMPLING_METRICS_TIMEOUT: '300'
  WAITING_TIME: '300'
  SAMPLE_SIZE: '0.3334'
  BAYESIAN: 'True'
  # n_startup_jobs: # of jobs doing random search at begining of optimization
  N_STARTUP_JOBS: '10'
  # n_EI_candidades: number of config samples draw before select the best. lower number encourages exploration
  N_EI_CANDIDATES: '24'
  # gamma: p(y) in p(y|x) = p(x|y) * p(x)/p(y) or specifically  1/(gamma + g(x)/l(x)(1-gamma))
  GAMMA: '0.25'
  NUMBER_ITERATIONS: '50'
  ITERATIONS_BEFORE_REINFORCE: '10'
  MAX_N_ITERATION_NO_IMPROVEMENT: '150'
  TRY_BEST_AT_EVERY: '1'
  RESTART_TRIGGER: '3' # https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
  REINFORCEMENT_RATIO: '0.3'
  PROBATION_RATIO: '0.3'
  METRIC_THRESHOLD: '0.0'
  RANDOM_SEED: '31'
  AGGREGATION_FUNCTION: 'sum'
  THROUGHPUT_THRESHOLD: '10'
  QUANTILE: '1.0'
  NAMESPACE: 'quarkus'
  HPA_NAME: 'quarkus-service'
  PROXY_IMAGE: 'smarttuning/proxy'
---
```
###### SmartTuning Deployment

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smarttuning
  namespace: quarkus
  labels:
    app: smarttuning
spec:
  template:
    metadata:
      name: smarttuning
      labels:
        app: smarttuning
    spec:
      containers:
        - name: smarttuning
          image: quay.io/smarttuning/smarttuning:dev
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 9090
          envFrom:
            - configMapRef:
                name: smarttuning-config
            - configMapRef:
                name: smarttuning-sampling-prom-queries
          volumeMounts:
            - mountPath: /etc/sampler-config
              name: sampler-config
              readOnly: true
          securityContext:
            privileged: true
      volumes:
        - name: sampler-config
          configMap:
            name: smarttuning-sampler-config
            items:
              - key: "sampler.json"
                path: "sampler.json"
  selector:
    matchLabels:
      app: smarttuning

```

## Jmeter

In this walkthrough SmartTuning is configured to tune applications based on
their workloads. SmartTuning uses a mock classficator that observes the number
of simultaneous clients set to Jmeter and uses this data to classify the
incoming workload. The [Jmeter
driver](optimization/manifests/icse/quarkus/jmeter-manifests/jmeter/k8s/02-driver-job.yaml)
is responsible for continually changes the number of simultaneous clients in
Jmeter and consequently alter the application workload.


The SmartTuning parameters defined above (see the snipped below) configures which
is the paramter in the ConfigMap SmartTuning will observes to figure out what is the current
workload.

```
  JMETER_CFG_WORKLOAD: "JUSERS"
  JMETER_CM: "jmeter-cm"
```


The [Jmeter
driver](optimization/manifests/icse/quarkus/jmeter-manifests/jmeter/k8s/02-driver-job.yaml)
need that your configuration match with Jmeter config, i.e., its ConfigMap name
should be the same that `JMETER_CM` set in the SmartTuning configurations, and
the variable that holds the workload identifiers (`JMETER_CFG_WORKLOAD:
"JUSERS"`) should match with the driver configuration.

###### Jmeter Driver ConfigMap

```
...
    # in this statemant 'JUSERS' must match with 'JMETER_CFG_WORKLOAD' value in 
    # the SmartTuning configuration
   -d "{\"kind\":\"ConfigMap\",\"apiVersion\":\"v1\",\"data\":{\"JUSERS\":\"$NAME\"}}" \
...
   # and the value of this variable must much with 'JMETER_CM' value in the
   # the SmartTuning configuration
   - name: CONFIGMAP_TARGET

```

### Summary

To quickly deploy this application you can run the [deployment
script](./optimization/manifests/icse/quarkus/deployment.sh) the application,
SmartTuning, and Jmeter in the correct order.


## Visualizing Logs

There are few different ways to visualize the SmartTuning execution.

##### stdout

Full logs (DEBUG and INFO) of the SmartTuning execution

`kubectl logs -f <smarttuning pod>`

or for a reduced scope and only tracing the major steps

`kubectl logs -f <smarttuning pod> | grep SMART_TUNING`

##### Mongo

To visualize the snapshot of SmartTuning at every iteration, access Mongo
through Mongo-Express

`kubectl port-forward service/mongo-workload-service  8081` and go to

`localhost:8081` to see the snapshots saved to the current experiment.


