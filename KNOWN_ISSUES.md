## Known Issues

* Due to concurrency issues, Smart Tuning may crash after a fresh deployment
  when trying to create a training replica. Re-deploy Smart Tuning if this
  happen.

* Smart Tuning cannot parse all JVM's if they are exposed through a
  `jvm.options` file. The current parser only handles the following parameters:
  * `-Xmns, -Xmnx, -Xmn, -Xmx, -Xms, -XX:SharedCacheHardLimit, -Xscmx,
    -Xtune:virtualized, -Xnojit, -Xnoaot`
  * `container_support=[true|false] --> -XX:[+|-]UseContainerSupport`
  * `gc_policy=[gencon|balanced|metronome|optavgpause|nogc] -->
    -Xgcpolicy:[gencon|balanced|metronome|optavgpause|nogc]`

* The controller responsible for create a training replica doesn't follows the
  best practices for controlling Kubernetes. It should be rewritten using
  [Koft](https://kopf.readthedocs.io/en/stable/), if intended to keep this
  python implementation, or either [Kubebuilder](https://book.kubebuilder.io/)
  or [Operator Framework](https://operatorframework.io/) for Go (language that I
  recommend to facilitate handling with K8s resources).

* The sampling module is duplicated. For metrics analysis and score calculation,
  Smart Tuning is using the latest and most modern version
  [metric2.py](optimization/smarttuning/models/metric2.py). However for workload
  classification Smart Tuning uses the outdated version
  [sampler.py](optimization/smarttuning/sampler.py)
