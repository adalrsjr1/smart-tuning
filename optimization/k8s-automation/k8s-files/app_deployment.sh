#!/bin/bash
kubectl apply -f prometheus/
kubectl apply -f acmeair-config.yaml
kubectl apply -f acmeair-service.yaml
kubectl apply -f acmeair-db-deployment.yaml
kubectl apply -f acmeair-smarttuning-tuning-prod.yaml
kubectl apply -f acmeair-smarttuning-tuning.yaml