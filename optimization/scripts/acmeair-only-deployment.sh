#!/bin/bash
# https://github.com/stakater/Reloader
kubectl --kubeconfig remote-config apply -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 10
kubectl --kubeconfig remote-config apply -f ../prometheus/
kubectl --kubeconfig remote-config apply -f ../manifests/acmeair/acmeair-config.yaml
kubectl --kubeconfig remote-config apply -f ../manifests/acmeair/acmeair-service.yaml
kubectl --kubeconfig remote-config apply -f ../manifests/acmeair/acmeair-db-deployment.yaml
kubectl --kubeconfig remote-config apply -f ../manifests/acmeair/acmeair-smarttuning-tuning-prod.yaml
kubectl --kubeconfig remote-config apply -f ../manifests/acmeair/acmeair-smarttuning-tuning.yaml




