#!/bin/bash

kubectl --kubeconfig=$HOME/.kube/$1 apply -f daytrader-ns.yaml
kubectl --kubeconfig=$HOME/.kube/$1 apply -f .

sleep 60

kubectl --kubeconfig=$HOME/.kube/$1 apply -f ./jmeter-manifests/workloads/driver/
kubectl --kubeconfig=$HOME/.kube/$1 apply -f ./jmeter-manifests/workloads/

sleep 10

kubectl --kubeconfig=$HOME/.kube/$1 apply -f ./smarttuning/

sleep 30
kubectl --kubeconfig=$HOME/.kube/$1 apply -f ./search-space
kubectl --kubeconfig=$HOME/.kube/$1 delete svc daytrader-smarttuning -n daytrader-jspjsfw
