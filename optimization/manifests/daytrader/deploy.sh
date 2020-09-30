#!/bin/bash

kubectl --kubeconfig=$HOME/.kube/trxrhel7perf/config apply -f daytrader-db.*.yaml
sleep 360
kubectl --kubeconfig=$HOME/.kube/trxrhel7perf/config apply -f *.yaml
