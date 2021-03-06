## Quick links

* [Repository structure](optimization/README.md)
* [Algorithm](optimization/docs/README.md)
* [Deployment](optimization/manifests/README.md)
* [Walkthrough](WALKTHROUGH.md)
* [Implementation](optimization/smarttuning/README.md)
* [Known Issues](KNOWN_ISSUES.md)

## Using Kubernetes

In case your are using Kubernetes through [Kind](https://kind.sigs.k8s.io/) and
this is installed remotely, you need to follow these steps:

1. create a cluster with kind
2. let `kubectl proxy` running in your remote node
3. create a ssh-tunnel to your remote node at port 8001: `ssh -N -L 8001:127.0.0.1:8001`
4. access any service running in kubernetes using this url template: `http://localhost:8001/api/v1/namespaces/<namespace>/services/<service-name>:8081/proxy/`. For more details refer to <https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/>

## Build

Run `make build` in [SmartTuning](optimization/) directory.

### Applications tested

* [AcmeAir](https://github.com/adalrsjr1/acmeair-monolithic-java)
* [Daytrader](https://github.com/adalrsjr1/sample.daytrader8)
* [Quarkus Demo](https://github.com/adalrsjr1/quarkusRestCrudDemo)

## Release notes

### Version 4.0
* smarttuning architecture relies on a state machine to progress over different
  tuning stages
* tuning multi-replicas services [#21](/../../issues/21)
* classify workloads based on the number of replicas
* classify workloads based on throughput
* add mock workload-classifier
* remove proxy need and service replication
#### Version 3.0
* update codebase to create one tuning context per workload [#18](/../../issues/18)
  * add eager stop
  * add option to turn on/off eager stop on poor configs
  * add pruning mechanism to eager stop a configuration sampled to another
    worklaod
  * add scheduler to switch tuning contexts and workload changes
* update codebase to use Optuna [#17](/../../issues/17)
* add new metrics to be used on objective function
  [#d2de64](https://github.ibm.com/Adalberto/smart-tuning/pull/17/commits/d2de64ef49e0a5b768fd4f7e24fb9a46040871d7), [#5d8deb](https://github.ibm.com/Adalberto/smart-tuning/pull/17/commits/5d8deb56a06aeb9276e36e51c52d31fc659aefe6)
* change trigger for restarting replicas to use mean+stddev rather median
  [#67ff0c](https://github.ibm.com/Adalberto/smart-tuning/commit/d2de64ef49e0a5b768fd4f7e24fb9a46040871d7)

#### Version 2.0
* ability to tune multiple pods at once
* quick-abortion of poor configs  [#1](/../../issues/1)
* set boundaries of dependent parameters on-the-fly [#10](/../../issues/10)
* uses two-phase algorithm to update a pod [#13](/../../issues/13)
