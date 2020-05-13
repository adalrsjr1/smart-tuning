#!/bin/bash

kubectl delete -f acmeair-config.yaml
kubectl delete -f acmeair-service.yaml
kubectl delete -f acmeair-db-deployment.yaml
kubectl delete -f acmeair-smarttuning-tuning-prod.yaml
kubectl delete -f acmeair-smarttuning-tuning.yaml
kubectl delete -f prometheus/
