#!/bin/bash

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f acmeair-ns.yaml
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f db/
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f .

sleep 60

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f jmeter-manifests/k8s

sleep 10

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f smarttuning/

sleep 30

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f search-space/

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 delete svc acmeair-svc-smarttuning -n acmeair



