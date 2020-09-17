#!/bin/bash
kubectl apply --kubeconfig=/Users/adalbertoibm.com/.kube/trxrhel7perf-1/config -f  https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 1
kubectl apply --kubeconfig=/Users/adalbertoibm.com/.kube/trxrhel7perf-1/config -f  ../prometheus
sleep 1
kubectl apply --kubeconfig=/Users/adalbertoibm.com/.kube/trxrhel7perf-1/config -f  ../grafana
sleep 1
kubectl apply --kubeconfig=/Users/adalbertoibm.com/.kube/trxrhel7perf-1/config -f  mongo-deployment.yaml
sleep 1
kubectl apply --kubeconfig=/Users/adalbertoibm.com/.kube/trxrhel7perf-1/config -f  search-space/search-space-crd-2.yaml
sleep 1
kubectl apply --kubeconfig=/Users/adalbertoibm.com/.kube/trxrhel7perf-1/config -f  tuning-deployment.yaml
sleep 1
kubectl apply --kubeconfig=/Users/adalbertoibm.com/.kube/trxrhel7perf-1/config -f  proxy-config.yaml
sleep 1
