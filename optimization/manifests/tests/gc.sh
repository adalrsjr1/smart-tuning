#!/bin/bash

kubectl delete deployment.apps nginx-test
kubectl delete deployment.apps nginx-test-smarttuning

kubectl delete svc nginx-test-svc
kubectl delete svc nginx-test-svc-smarttuning

kubectl delete cm nginx-test-cm
kubectl delete cm nginx-test-cm-2
kubectl delete cm nginx-test-cm-smarttuning
kubectl delete cm nginx-test-cm-2-smarttuning
