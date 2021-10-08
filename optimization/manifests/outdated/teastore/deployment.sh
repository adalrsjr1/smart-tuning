#!/bin/bash

kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f 00-teastore-all.yaml
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f db/
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f jmeter/k8s/01-jmeter-cm.yaml
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f smarttuning/

sleep 10
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f .
sleep 30
kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f search-space/
sleep 120


kubectl --kubeconfig=/Users/adalrsjr1/.kube/$1 apply -f jmeter/k8s/
