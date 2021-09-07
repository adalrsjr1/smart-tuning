#!/bin/bash

kubectl --kubeconfig=$HOME/.kube/$1 apply -f daytrader-ns.yaml
kubectl --kubeconfig=$HOME/.kube/$1 apply -f ./jmeter-manifests/workloads/driver/
kubectl --kubeconfig=$HOME/.kube/$1 apply -f ./smarttuning/
kubectl --kubeconfig=$HOME/.kube/$1 apply -f .
sleep 30
kubectl --kubeconfig=$HOME/.kube/$1 apply -f ./search-space
sleep 90
kubectl --kubeconfig=$HOME/.kube/$1 apply -f ./jmeter-manifests/workloads/

