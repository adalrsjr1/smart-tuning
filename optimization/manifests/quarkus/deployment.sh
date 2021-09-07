#!/bin/bash

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f quarkus-ns.yaml
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f db/
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f .

sleep 60

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f jmeter-manifests/jmeter/k8s

sleep 10

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f smarttuning/

sleep 30

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f search-space/

