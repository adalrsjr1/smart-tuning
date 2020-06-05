#!/bin/bash
kubectl delete -f acmeair-config.yaml
kubectl delete -f acmeair-service.yaml
kubectl delete -f mongo-deployment.yaml
kubectl delete -f acmeair-db-deployment.yaml
kubectl delete -f tuning-deployment.yaml
sleep 10
#kubectl delete -f acmeair-smarttuning-sync.yaml
kubectl delete -f acmeair-smarttuning-tuning-prod.yaml
sleep 10
kubectl delete -f acmeair-smarttuning-tuning.yaml
kubectl delete -f ../jmeter_k8s.yaml
# https://github.com/stakater/Reloader
kubectl delete -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
