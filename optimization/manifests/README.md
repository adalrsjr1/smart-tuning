## Set up environment

`$ ./monitoring-deploy.sh $HOME/.kube/config`

Be careful with `limits` set in deployments, e.g.,  Prometheus. If your
Kubernetes cluster has limited resources, you can either downsize or delete the
limits in such deployments.

### MongoDB

The Mongo version in `mongo-deployment.yaml` is a hard requirement. If using a
newer version, SmartTuning will not save the logs properly.

## Search Space

`search-space-cdr-2.yaml` is the latest version for the Search Space CDR.
SmartTuning uses the specification of a CDR to defines the search space during
the tuning.

### Search Space Scheme

```
spec:
  deployment: [string]
  service: [string]
  namespace: [string]
  manifests:
    - name
      type: [deployment|configMap]
data:
  # each item in this data list defines a group of option or number tunables
  - name: [a manifest name as specified before]
    tunables:
      option: # option tunable defines a tunable with a well defined list of
              # values, e.g., gc policies or discrete memory values [512, 1024,
              4096]
        - name: [string] # this must match with the envVar name
          type: [real|integer|string]
          values:
            - "1"
            - ...
      number:
        - name: [string] # this must match with the envvar name
          lower:
            value: [any numeric value less than upper]
            dependsOn: [string] # the name of other tunable that limits this tunable
                                # e.g. - name: -Xms
                                #         lower:
                                #          value: 128
                                #          dependsOn: -Xmx 0.8 *
                                # note that dependsOn expression must be in [RPN](https://en.wikipedia.org/wiki/Reverse_Polish_notation) notation
                                # also avoid cyclic dependencies a->b->a
          upper:
            value: [any numeric value greater than lower]
            dependsOn: [string]
          step: [any numeric value] # minimum distance between two sequencial values
          real: [bool] # informs if SmartTuning will sample floats or integers
```

## SmartTuning

```
00-smarttuning-sampler-config.yaml
tuning-cm.yaml
tuning-deployment.yaml
tuning-rabc.yaml
tuning-svc.yaml
```
#### 00-smarttuning-sampler-config.yaml

This `ConfigMap` defines the queries for sampling metrics values and the
objective and penalization functions.

```
data:
  sampler.json: |
  {
    "objective": - (penalization) * throughput/memory
    "penalization": "1/(1+max(0, 100*(memory/meory_limit)-79.000))
    # any metric specified in 'metrics' can be used to write an objective or
    penalization function
    # use the metric 'name' to refer to a metric
    "metrics": [
      {
        "name": [string] # any name you choose
        "query": [string] # query for sampling a metric value, see examples
        below
        "datasource": [prom|cfg|scalar] # see examples below
      },
      {
        "name": "memory",
        # all queries are interpreted in Python and then sent to the datasource
        # therefore, it is necessary scaping some chars like { and ". Furthermore,
        # the application pod name and its namespace are referencied as
        # {podname}- and {namespace} respectively. Finally, {interval} is
        # specified regarding the iteration lenght, and so must be just referencied
        # here when needed.
        #
        # all prometheus queries need to be in this specific format, heads up
        # with the need of a '-' (dash) after {podname}, e.g, '{podname}-'.
        # and '"' (double quotes) need to be scaped, e.g., \", as well as
        # the '{}' (curly braces), that are scaped being double, e.g, {{}}.
        #
        "query": "f'sum(max_over_time(container_memory_working_set_bytes{{id=~\".*kubepods.*\",pod=~\"{podname}-.*\",name!~\".*POD.*\",namespace=\"{namespace}\",container=\"\"}}[{interval}s]))'"
        "datasource": "prom"
      },
      ...
      {
        "name": "jdbc_max_conn",
        # for observing the values in envVars, uses the follwing format:
        # <configMap.metadata.name>.<envVar name>, see below: 
        "query": "daytrader-config-jvm.-Xmx",
        "datasource": "cfg"
      },
      {
        "name": "custom_scalar",
        "query": "3.14"
        "datasource": "scalar"
      }
    ]
  }
```

#### tuning-cm.yaml

This file defines two `configMaps`: `smarttuning-sampling-prom-queries` and
`smarttuning-config`.

`smarttuning-sampling-prom-queries` exists exclusively for
workload classification if classifying based on observed volume at runtime.
Otherwise, these queries are useless. Queries for all metrics as specified for
historical reasons.

`smarttuning-config` configures SmartTuning for tuning an application.

```
data
  # uses a custom proxy or not. If False, updates the queries in 00-smarttuning-sampler-config.yaml
  # to use the queries from this proxy (smarttuning_http_requests_total,
  smarttuning_http_processtime_seconds_sum)
  NO_PROXY: "True"
  # ST logging level NOTSET activate all levels, SMART_TUNING is a reduced
  # version of INFO with a nice output formated. Other options are: INFO, DEBUG,
  ERROR, WARNING -- always upper case
  LOGGING_LEVEL: "NOTSET"
  # file specified in 00-smarttuning-sampler-config.yaml
  SAMPLER_CONFIG: "/etc/sampler-config/sampler.json"
  # create an exclusive service for training replica, if true it is necessary
  # manually poiting the clients to hit this replica
  TWO_SERVICES: "False"
  # dummy workload classifieers:
  # HPA: based on number of replicas
  # RPS: based on volume of requests (depends on queries at smarttuning-sampling-prom-queries) 
  # based on the value of JMETER_CFG_WORKLOAD at JMETER_CM
  WORKLOAD_CLASSIFIER: "CM" #HPA or RPS or CM
  JMETER_CFG_WORKLOAD: "JUSERS"
  JMETER_CM: "jmeter-cm"
  # if using WORKLOAD_CLASSIFIER: RPS this var specifie in how many bands
  # this classifier will classifie the worklaods. The values specified, defines
  # the center point of the band
  WORKLOAD_BANDS: "50, 110, 170"
  # quickly stop an iteration if workload changes -- not reliable
  FAIL_FAST: "False"
  # print this config file imediately when deployed 
  # kubectl logs smarttuning-pod-name | head -n 30
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
  # iteration lenght in seconds
  WAITING_TIME: '300'
  # interation subinterval (WAITING_TIME * SAMPLE_SIZE)
  SAMPLE_SIZE: '0.3334'
  # baysian (True) or random sampling (False)
  BAYESIAN: 'True'
  # n_startup_jobs: # of jobs doing random search at begining of optimization
  N_STARTUP_JOBS: '10'
  # n_EI_candidades: number of config samples drawn and evaluated in the
  # surrogate function before select the best and apply in training replica
  N_EI_CANDIDATES: '24'
  # gamma: p(y) in p(y|x) = p(x|y) * p(x)/p(y) or specifically  1/(gamma + g(x)/l(x)(1-gamma))
  GAMMA: '0.25'
  # max number of iterations
  NUMBER_ITERATIONS: '50'
  ITERATIONS_BEFORE_REINFORCE: '10'
  # number of reinforcement iterations: ITERATIONS_BEFORE_REINFORCE *
  REINFORCEMENT_RATIO
  REINFORCEMENT_RATIO: '0.3'
  # same as before, but for probation
  PROBATION_RATIO: '0.3'
  # allows to run two experiment with the same configuration sampled
  RANDOM_SEED: '31'
  # application namespaces -- by design one single instance of SmartTuning can
  # tune applications in different namespaces but this feature become outdated
  # over time
  NAMESPACE: 'quarkus'
  # HPA name that handles the application replicas. SmartTuning will start only
  # when it observes events from this component.
  HPA_NAME: 'quarkus-service'
  # proxy image name if using the custom made proxy
  PROXY_IMAGE: 'smarttuning/proxy'

```

#### Other manifests

The manifests `tuning-deployment.yaml, tuning-rabc.yaml, tuning-svc.yaml` are self explanatory.

## Applications

All applications in `icse` folder has valid manifests and running
`deployment.sh` will deploy then into your local kubernetes. To deploy them in a
remote K8s, edit this script accordingly with the correct `kubeconfig`.

### Application requirements

For SmartTuning tune an application, this application must expose their
parameters through `envVar`s specified into `configMaps`. These `configMaps`
must be defined in the `search-space` of the application. Furthermore, the
application deployment must specify the memory and cpu limits, and this
deployment name must also be defined in the `search-space`.

Both applications' deployment and service manifest must be annotated with
`injection.smarttuning.ibm.com: "true"` as below:

```
metadata:
  labels:
    app: quarkus-service
  annotations:
    injection.smarttuning.ibm.com: "true"
  name: quarkus-service
  namespace: quarkus
```

If the application as longer warm-up times, it is recommended to set [start up
and readiness probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/),
and set a `PreStop` and `terminationGracePeriodSeconds` to avoid long warm up
times with the application doesn't receiving any requests -- see daytrader
deployment manifest as example. Below is the requirements to avoid this warm-up
issue in most of cases.

```
...
kind: Deployment
...
spec:
...
  strategy:
    type: RollingUpdate
      maxSurge: "100%"
      maxUnavailable: "0%"
terminationGracePeriodSeconds: 90
...
containers:
  - name: ...
    lifecycle:
      preStop:
        exec:
          command: ["/bin/bash", "-c", "sleep 90"]
...
```

## Running

To run an experiment just run `icse/<folder name>/deployment.sh` and it will
deploy an application + smart tuning + jmeter. The deployment order in this
script must be respected for any deployment.
