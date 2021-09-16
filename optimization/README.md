# Repository structure
```
optimization/
  |_/docs
  |_/experiments
  |_/grafana
  |_/kube-state-metrics
  |_/manifests
  |_/metrics-server
  |_/prometheus
  |_/proxy
  |_/reloader
  |_remote-config
  |_/smarttuning
  |_/scripts

```
## Docs

This folder has SmartTuning's architecture diagram and other related
documentation.

## Experiments

This folder has the script `plot_2phase_planner_new.py` for plotting all data
SmartTuning saves into MongoDB during its execution.

## Grafana

This folder has the manifest for deploying Grafana and several dashboards to
visualize the execution of applications being tuned. `smarttuning-replicas.json`
is latest dashboard created.

## Kube-state-metrics and metrics-server

Local repositories for `kube-state-metrics` and `metrics-server`.

## Manifests

```
manifests/
  |_/icse
  |_/search-space
```

The scripts `/manifests/monitoring-deploy.sh` and
`/manifests/monitoring-delete.sh` install and uninstall all plumbing necessary
for Smart Tuning. The `/manifests/icse` folder has the manifests necessary to
  deploy benchmarks applications and Smart Tuning on Kubernetes. To deploy any
  of these applications, just run `deployment.sh` and it will install all into
  your local Kubernetes. Finally, the other folders in `manifests/` are kept for
  historical reasons.


## Prometheus

This folder contains all manifests necessary for deploying Prometheus.

## Proxy

`Proxy` is a custom made proxy implemented to observe requests flowing into the
application. For using this, mark the SmartTuning's envVar `NO_PROXY=False` and
SmartTuning automatically will inject this proxy in the application Pod. If
using this proxy, the queries for sampling metrics should be updated
accordingly. See deployment for more.

## Reload

This folder is kept for historical reasons. It has a custom made implementation
for reload pods when their `ConfigMaps` are updated. This implementation was
  replaced in favor of
  [stakater/Reloader](https://github.com/stakater/Reloader).

## SmartTuning

This folder has the SmartTuning implementation.

# Quick Start

1. run `manifests/monitoring-deploy.sh`
  1. install a proper dashboard to grafana
2. deploy the application `manifests/icse/quarkus/deployment.sh`
3. monitors SmartTuning
  1. `kubectl -n kube-monitoring port-forward service/grafana-service 3000` and access `localhost:3000`
  2. `kubectl port-forward service/mongo-workload-service  8081` and access `localhost:8081` for detailed logs.
