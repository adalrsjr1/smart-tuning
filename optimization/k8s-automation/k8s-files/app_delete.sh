#!/bin/bash
kubectl --kubeconfig remote-config delete -f acmeair-config.yaml
kubectl --kubeconfig remote-config delete -f acmeair-service.yaml
kubectl --kubeconfig remote-config delete -f acmeair-db-deployment.yaml
kubectl --kubeconfig remote-config delete -f acmeair-smarttuning-tuning-prod.yaml
kubectl --kubeconfig remote-config delete -f acmeair-smarttuning-tuning.yaml
# https://github.com/stakater/Reloader
kubectl --kubeconfig remote-config delete -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
