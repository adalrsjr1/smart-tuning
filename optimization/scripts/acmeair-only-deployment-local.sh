#!/bin/bash
# https://github.com/stakater/Reloader
kubectl apply -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 10
kubectl apply -f ../prometheus/
kubectl apply -f ../manifests/acmeair/acmeair-config.yaml
kubectl apply -f ../manifests/acmeair/acmeair-service.yaml
kubectl apply -f ../manifests/acmeair/acmeair-db-deployment.yaml
kubectl apply -f ../manifests/acmeair/acmeair-smarttuning-tuning-prod.yaml
kubectl apply -f ../manifests/acmeair/acmeair-smarttuning-tuning.yaml




