# Deploying SmartTuning

```
optimization/
  |_/docs
  |_/experiments
  |_/grafana
  |_/manifests
  |_/prometheus
  |_/proxy
  |_/reloader
  |_remote-config
  |_/smarttuning
  |_/scripts

```

## Manifests

This folder has all manifests files necessary to deploy the `AcmeAir` or
`Daytrader` apps.

### AcmeAir and Daytrader

[AcmeAir Repo](https://github.com/blueperf/acmeair-mainservice-java)


[Daytrader Repo](https://github.com/OpenLiberty/sample.daytrader8)

The `*.config.yaml` are the configuration files to set the applications app. If
the `*.config.yaml` has some entry named `*jvm*` this are the configurations
related to JVM (OpenJ9) otherwise they are the configuration specific to the application
or application server (Open Liberty).

The `search-space` folder has the K8s CRD used to limit the values that
SmartTuning can sample to try a new configuration. The `search-space` definition
is in [search-space/search-space-crd-2.yaml](../search-space/).

The `jmeter` folder has the manifest necessary to deploy Jmeter performance test
client. When deploying Jmeter there are only two key configuration that may be
adjusted:

* JTHREAD: '200' -- The number of threads (virtual clients) used during the
  experiment
* JDURATION: '15000' -- The duration of the experiment in seconds.

### SmartTuning

This folder has the manifests related to SmartTuning deployment. Below are the
configurations that likely will be changed between the applications

* `SAMPLING_METRICS_TIMEOUT: '1200'` -- The interval experimenting a configuration
* `WAITING_TIME: '1200'` -- The interval waiting between each deployment.
  `SAMPLING_METRICS_TIMEOUT` and `WAITING_TIME` should be set the same.
* `SAMPLE_SIZE: '0.3334'` -- The interval used to compare two metrics. E.g.,
  0.3334 with sample the last 3rd part of the experimenting interval
  (WAITING_TIME) of the training and production pod. SmartTuning uses this value
  sampled to compare whether training pod has a better performance than the
  production pod.
* `K: '1'`: -- The max number of workload types that SmartTuning can classify
* `URL_SIMILARITY_THRESHOLD: "0.1"` -- Similarity threshold that SmartTuning uses
  to compare two URLs
* `OBJECTIVE: '-(throughput / ((((memory / (2**20)) * 0.013375) + (cpu * 0.0535) ) / 2))'` -- Objective function to optimize the application
* `POD_REGEX: 'daytrader-.*servicesmarttuning-.+'` -- Training pod regex name
* `POD_PROD_REGEX: 'daytrader-.*service-.+'` -- Production pod regex name

### Other files

`mongo-deployment.yaml` is the manifest to deploy the instance of Mongo that
SmartTuning uses to log the experiments iterations. It is possible to inspect
it accessing `http://localhost:30081`.

`prox-config.yaml` is the file with the proxy configurations.

`monitoring-*.sh` are the scripts to deploy the monitoring machinary on K8s
(Prometheus and Grafana), the pod reloader, and the search-space CRD. See next section.

## Quick Start

1. run `manifests/monitoring-deploy.sh`
2. deploy the application `manifest/acmeair` or `manifest/daytrader` -- be
   careful with the DB deployments
3. deploy SmartTuning from the application folder
4. deploy jmeter from the application folder
5. deploy the search-space from the application folder

Access the Grafana's dashboard `http://localhost:30030` and import the grafana
dashboard json from `../grafana/smarttuning.json`


