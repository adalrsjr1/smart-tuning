#!/bin/bash
kubectl --kubeconfig=remote-config delete -f acmeair-config.yaml
kubectl --kubeconfig=remote-config delete -f acmeair-service.yaml
kubectl --kubeconfig=remote-config delete -f mongo-deployment.yaml
kubectl --kubeconfig=remote-config delete -f acmeair-db-deployment.yaml
kubectl --kubeconfig=remote-config delete -f tuning-deployment.yaml
sleep 10
#kubectl --kubeconfig=remote-config delete -f acmeair-smarttuning-sync.yaml
kubectl --kubeconfig=remote-config delete -f acmeair-smarttuning-tuning-prod.yaml
sleep 10
kubectl --kubeconfig=remote-config delete -f acmeair-smarttuning-tuning.yaml
kubectl --kubeconfig=remote-config delete -f ../jmeter_k8s.yaml
# https://github.com/stakater/Reloader
kubectl --kubeconfig=remote-config delete -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
kubectl --kubeconfig=remote-config delete -f manifests/search-space/
