#!/bin/bash
kubectl --kubeconfig remote-config delete -f ../manifests/acmeair/acmeair-config.yaml
kubectl --kubeconfig remote-config delete -f ../manifests/acmeair/acmeair-service.yaml
kubectl --kubeconfig remote-config delete -f ../manifests/acmeair/acmeair-db-deployment.yaml
kubectl --kubeconfig remote-config delete -f ../manifests/acmeair/acmeair-smarttuning-tuning-prod.yaml
kubectl --kubeconfig remote-config delete -f ../manifests/acmeair/acmeair-smarttuning-tuning.yaml
# https://github.com/stakater/Reloader
kubectl --kubeconfig remote-config delete -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
