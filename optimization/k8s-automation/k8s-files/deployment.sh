#!/bin/bash
kubectl apply -f prometheus/
kubectl apply -f acmeair-config.yaml
kubectl apply -f acmeair-service.yaml
kubectl apply -f mongo-deployment.yaml
kubectl apply -f acmeair-db-deployment.yaml
kubectl apply -f tuning-deployment.yaml
sleep 10
#kubectl apply -f acmeair-smarttuning-sync.yaml
kubectl apply -f acmeair-smarttuning-tuning-prod.yaml
sleep 10
kubectl apply -f acmeair-smarttuning-tuning.yaml
