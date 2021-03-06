#!/bin/bash

kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f ./smarttuning/00-smarttuning-sampler-config.yaml -f ./smarttuning/tuning-deployment-no_proxy.yaml
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f .
sleep 5
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f ./search-space/daytrader-ss-replicas-no_proxy.yaml
sleep 120
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f ./jmeter-manifests/workloads/driver/01-jmeter-cm.yaml -f ./jmeter-manifests/workloads/single_dyn_jmeter_k8s_prod.yaml
