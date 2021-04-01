#!/bin/bash


kubectl --kubeconfig=$HOME/.kube/trinity01/config delete deployment daytrader-servicesmarttuning

kubectl --kubeconfig=$HOME/.kube/trinity01/config delete cm daytrader-config-appsmarttuning daytrader-config-jvmsmarttuning


kubectl --kubeconfig=$HOME/.kube/trinity01/config delete -f .
kubectl --kubeconfig=$HOME/.kube/trinity01/config delete -f ./search-space
kubectl --kubeconfig=$HOME/.kube/trinity01/config delete -f ./jmeter-manifests/workloads/driver/01-jmeter-cm.yaml

kubectl --kubeconfig=$HOME/.kube/trinity01/config delete -f ./jmeter-manifests/workloads/
kubectl --kubeconfig=$HOME/.kube/trinity01/config delete -f ./smarttuning

