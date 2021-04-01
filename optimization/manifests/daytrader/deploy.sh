#!/bin/bash

kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f ./smarttuning
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f .
sleep 5
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f ./search-space/daytrader-ss-replicas.yaml
sleep 70
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f ./jmeter-manifests/workloads/driver/01-jmeter-cm.yaml -f ./jmeter-manifests/workloads/driver/01-jmeter-cm-train.yaml

kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f ./jmeter-manifests/workloads



