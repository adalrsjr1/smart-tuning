#!/bin/bash
kubectl delete -f ../manifests/mongo-deployment.yaml
kubectl delete -f ../manifests/tuning-deployment.yaml
kubectl delete -f ../manifests/acmeair/acmeair-config.yaml
kubectl delete -f ../manifests/acmeair/acmeair-service.yaml
kubectl delete -f ../manifests/acmeair/acmeair-db-deployment.yaml
kubectl delete -f ../manifests/acmeair/acmeair-smarttuning-tuning-prod.yaml
kubectl delete -f ../manifests/acmeair/acmeair-smarttuning-tuning.yaml
kubectl delete -f ../manifests/jmeter/jmeter_k8s.yaml
# https://github.com/stakater/Reloader
kubectl delete -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
