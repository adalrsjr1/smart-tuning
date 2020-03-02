# Introduction

The deployment of Cloud-Native applications (CNA), a.k.a. microservices-based applications, relies on placing a
collection of small, independent, loosely coupled and, ideally, self-managed services into a cluster. This design
facilitates a decoupled evolution of the application's components and brings reliability, due to decoupled failures
that can be isolated to avoid the crash the whole application. However, the decoupling makes the application management
difficulty, considering the large number of distributed pieces of software that engineers should handle.

Consequently, these services are mostly bundled up with several runtime abstraction layers to facilitate the
application execution or its runtime management. Some examples of these layers are application servers (e.g.,
OpenLiberty), runtime environment (e.g., Java Virtual Machine), hardware isolation (e.g., containers and virtual
machines), and cluster orchestrator (e.g., Kubernetes).

Runtime abstractions layers are general purpose with many parameters (knobs), that the application's engineer adjusts
to better satisfy the use cases of the application. Many of these knobs are related to data structures or
infrastructure mechanisms, so their adjustment affects the application's performance. An important issue regarding
these knobs is the colossal number of possibilities that an engineer has available.

Every layer added increases exponentially the number of possible configurations that the engineer has available to
experiment. Making matters worse, an optimal configuration for an application is mutable, it should changes according
to the environment, such as fluctuations of number of users requests, saturation of the cluster, version of libraries
being used, and so on. For tuning an application the engineer has to evaluate many different configurations to find one
that extracts the most of the application performance for a given situation.

To tackle this challenge, we are proposing SmartTuning, a mechanism to automatically identify and apply optimal
configurations onto CNA regarding the environment where they are deployed. The rational of this mechanism is, based on
changes of the application workload (incoming requests and resources consumption) overtime, find out, and continually
updates, a (quasi-) optimal configuration which extracts the maximum performance of the application.

# Design Overview

How SmartTuning improves an application unfolds in two main steps: find out optimal configurations to a
given application's behaviour, and identify when application's behaviour changes. Hence, SmartTuning can associate an
optimal configuration to different behavior, making the application works always in its best performance.



The overall ideal if SmartTuning is observes the behavior of the application over time and identify patterns, such as
which intervals the application has high resources consumption and which is the incoming traffic to the application in
this interval. In parallel SmartTuning search for an optimal configuration to improve the per

# Assumptions

In order to find an optimal, or at least quasi-optimal, configuration for an application we made some assumptions.
First of them, we are modeling applications as a black-box functions of performance over time. We assume that along a
long time interval the application has different performances, and this performances are recurrent, so that similar
behaviours of the application repeats in timed intervals. For instance, every day in a same hour the application has a
similar performance due to a recurrent access pattern of its users.

Also, we assuming that the behaviour of the application remains long enough to be identified, evaluated, and improved.
I.e., the behaviour of the application changes few times on a given time interval.

We assuming that the traffic to the application can be modelled in a reduced scope. So that, measuring the incoming
traffic to a single replica we can generalize the traffic of the whole application.

We assuming that applications are properly instrumented and all kind of data can be monitored. To do so, metrics like
CPU and memory usage, and hits to API are measured. Measuring the hits to API is fundamental to identify the usage
pattern of the application, i.e., which components are exercised overtime.

 - single objective
 - 

# Challenges

# Long-term applications - deployments are rare to happen

## Deploying the application

1. Reset learning model
2. Two replicas

Initially, the tuning system resets the learning model, for it does not compute a mistuning. In this context, the
learning model is all computation already made and kept, such as the frequency of URLs hits patterns (histograms),
application configurations, and the workloads forecasting. When the engineer deploys the application, it is supposed
that some components have new requirements and/or use new technologies.  Thereby a new configuration should be computed
from scratch. For instance, a new deployment can change the API of the application, by adding or removing URLs, and all
histograms prior computed became invalidated. Therefore, the use of configurations before calculated and categorized on
a new deployment can lead the application to a poor performance state, or worst, it can crash the application.

> The tuning system reset the model for every deployment, so after
> days computing a (quasi-) optimal setup, the system puts away. It is
> not productive at all! How can we smartly deal with it? Is there
> anything that should remain in the learning model between
> deployments?

When the engineer deploys the application, the tuning system should
replicate the application automatically so that the tuning loop does
not threaten its performance in production. Let's assume that the app
is a stateless pod. Hence, the design of a cloud-native application
should consider horizontal auto-scaling inevitable. Therefore, it is
coherent to force the application to have a new replica for tuning
porpoises.

We are assuming that part of the application's traffic (25%) can model
the whole application's traffic so that we can use the subset of
requests in a replica to compute an optimal tuning.

> In a plain configuration of K8s, each replica has a uniform chance
> to receive a request (Random Load Balancing). Hence, to the tuning
> pod receives 25% of the requests, it is necessary 3 other pods in
> production (3 in production and 1 for tuning), giving us the
> probability of 25 of pods receive a request. The Openshift, on the
> other hand, allows weighted traffic natively, so it is trivial to
> send a specific amount of requests to a particular replica.

> In the worst case, behind a Round-Robin or Random load balancer, the
> requests received by the tuning-replica might not represent the
> whole workflow of the application. How can we guarantee that the
> subset of requests represents the entire application?

## Sampling Metrics

1. Sampling Metrics --> avg, std. dev.
2. Sampling URL hits --> histograms
3. Store workload context (histogram, (metrics), start, end)

All data necessary to compute the tuning, i.e., resources metrics and
application workflow (app's URLs hits), will continually be sampled
from the application in timed intervals, e.g., hourly or daily. A
resource metric sample is its average and standard deviation in the
past t time-unit, e.g., the average CPU load in the past hour.

> Regarding CPU metric. We should consider milicores over CPU load
> since milicores is the native CPU metric in a K8s environment.

For gathering the workflow, we are assuming that application keeps
counters for every URL hit in its API. Hence, the counters are sampled
in a given interval (like resources metrics) and aggregated into a
map, associating every URL (the base URL excluding query parameters)
to its counter. The asociation becomes a histogram of the URLs in an
interval (t1, t2).

The tuning system stores the resources metrics and histograms as
following:

```json
{
  "histogram": {
    "url1": 13,
    "url2": 8,
    "url3": 21,
		...
	},
	"cpu": {"avg": MILICORES, "std": MILICORES, "n_samples": N}
	"memory": {"avg": BYTES, "std": BYTES, "n_samples": N}
	"start": UNIX-TIMESTAMP,
	"end": UNIX-TIMESTAMP
}
```

## Computing configurations

Repeat indefinitely:

1. Compute an new configuration
2. Apply the configuration on the application
3. Check feature in past interval, e.g., throughput
4. Store configuration context (configuration, (feature:value), start, end)

It is trivial for the tuning system to compute a new configuration
through Bayesian or Random Search methods. Then, the tuning system
applies the setting the replica dedicated to the tuning
reconfigurations and we wait for t time-units before to check the
consequence of the configuration on the application. After the waiting
time, the application's performance is measured, such as throughput or
latency, and associated with the setting selected.

The configuration and the feature value are stored, both associated
with a time interval, in the structure as follows:

```json
{
  "configuration": { "param1": VALUE, "param2": VALUE, ...  },
  "feature": { "name": STRING, "value": NUMBER },
  "start":UNIX-TIMESTAMP,
  "end": UNIX-TIMESTAMP
}
```

> Bayesian Optimization may not be the best strategy to find an
> optimal configuration for real-world workloads. In a scenario where
> we have no control of how similar are the incoming workflows to the
> application under study, the sequence of different workflows will
> not may not reproduce the same behavior into the application. Hence,
> the Bayesian Optimization will behave just like a fancy random
> search.

> Bayesian Optimization is coherent to be used if, in an interval (t0,
> tk), every subinterval [(t0, t1), ... , (tk-1, tk)] has a histogram
> very similar. The application understudy will have the same input
> for all configurations bean evaluated. Thus the result may converge
> to the optimal after n iterations or less. However, the tuning
> system has no contol over how similar are the incoming workloads to
> the application. Hence, Bayesian Optimization cannot learn how to
> find a suitable configuration within the search space, which makes
> its inherited statistics useless.

> Therefore, a simple random search would enough to find (quasi-)
> optimal configurations. For a long enough running, eventually, the
> random search will find an optimal configuration, that may be as
> fast as Bayesian Optimization with different workloads inputs.

The configuration and the measured application performance, after
their calculation, are associated with the workload sampled.

## Workloads-classification

1. Group the workloads into types
2. Detect the best configuration for a type (centroid workload in a group)
3. Learn when a workload type come up to the application

The tuning system compares all-to-all the workloads (histograms and
metrics) after sampling. Only the setting that led the application to
the best performance remains when the workloads are equal, respecting
a given threshold.

> There are two main components in a workload vector (histograms,
> metrics) that are vectors themselves. They are orthogonal with
> totally different semantics. Therefore, it is necessary attention
> when comparing workloads.

> Histograms are same sized vectors which represent the URL-hit
> probability on application's API. Due to the "probability" in their
> definition, the tuning system must compare the histograms by using
> distribution comparison techniques, such as Hellinger distance.
> Hence the vectors must be normalized between 0-1. The semantics of
> this comparison means how similar are the shapes of histograms. A
> similar outcome would reach by Cosine distance, which does not need
> normalized vectors.
To compare the magnitude, i.e., the frequency of URL-hits, of the same
shape histograms, we can apply the simple Euclidean distance.

> We can use Z-Test to compare if other metrics are significantly
> different or not.

> Two workloads are equal if their histograms (shapes and magnitudes)
> and metrics are similar, respecting a given threshold. E.g.,
> h1.shape == h2.shape + threshold. The consequence of this approach
> is a large number of groups that may come up during the system
> execution.

In this step, every group of workloads has a configuration and a list
of timestamps (when the workloads happened) associated with it. Then,
the classifier uses this association to learn when to apply a
configuration.

> The classifier uses only the timestamp and the groups as input for
> the learning step. The classifier splits the timestamp in different
> granularities, such as hour, day, week, and month.

With all workloads (associated with a timestamp), we can learn how to
forecast the type of workload that will arrive in the application. So
we can preempt a configuration to the application given the current
time.

> The process of learning is updated synced with the groups. Changes
> in groups reinforce the learning process. Which learning algorithm
> should we use?

> How to handle situations where the incoming workload is unknown? For
> some reason, a new workload can come up, and none of the previous
> configurations knows how to address this workload adequately. So,
> how long should the system wait before it drops the setting applied
> to handle this unknown workload? -- the configuration can jeopardize
> the performance of the system. To avoid misconfiguration, it should
> exist a default configuration, hence for unknown workloads. The
> default configuration is applied.

> It is important to notice that, once a new workload comes up, the
> system can associate it to a (new) group, and after that, become
> ready to handle it in the next situation.


## Self-tuning

* Select a configuration based on time and apply to the application

The system has learned when the workloads show up, e.g., time of day
or day in a week, then it can apply configurations to tuning the
system automatically.

> For instance, during a day, the system learns which period every
> workload appears (day or night, or morning, afternoon, or night).
> Then, it has to determine which weekday the workloads appears and
> combine with the previous information, e.g., Monday mornings,
> Thursday nights, and so on. This idea is extrapolated to different
> granularities. Then, the system can infer that: *( **workload K** ) in
> nights before black-Friday's there is many requests looking for
> products*; and: *( **workload J** ) in weeks after black Friday there is
> more requests on post-sale support because there are several
> returning inqueries from costumers* . And for each of this situations
> should exists an (quasi-) optimal configuration/tuning.

# Short-term applications (updates are frequent)

## Deploy Application

Same as for long-term applications.

## Sampling Metrics

Same as for long-term applications.
## Compute configurations

TODO

## Workloads-classification

TODO
