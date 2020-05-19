#!/bin/bash
# https://github.com/stakater/Reloader
kubectl apply -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 10
kubectl apply -f prometheus/
kubectl apply -f acmeair-config.yaml
kubectl apply -f acmeair-service.yaml
kubectl apply -f mongo-deployment.yaml
kubectl apply -f acmeair-db-deployment.yaml
kubectl apply -f tuning-deployment.yaml
kubectl apply -f acmeair-smarttuning-tuning-prod.yaml
kubectl apply -f acmeair-smarttuning-tuning.yaml

echo 'waiting for application be read, then start jmeter'
sleep 60
kubectl apply -f ../jmeter_k8s.yaml
