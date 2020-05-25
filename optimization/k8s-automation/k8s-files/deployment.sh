#!/bin/bash
# https://github.com/stakater/Reloader
kubectl --kubeconfig=remote-config apply -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 10
kubectl --kubeconfig=remote-config apply -f prometheus/
kubectl --kubeconfig=remote-config apply -f acmeair-config.yaml
kubectl --kubeconfig=remote-config apply -f acmeair-service.yaml
kubectl --kubeconfig=remote-config apply -f mongo-deployment.yaml
kubectl --kubeconfig=remote-config apply -f acmeair-db-deployment.yaml
kubectl --kubeconfig=remote-config apply -f tuning-deployment.yaml
kubectl --kubeconfig=remote-config apply -f acmeair-smarttuning-tuning-prod.yaml
kubectl --kubeconfig=remote-config apply -f acmeair-smarttuning-tuning.yaml

echo 'waiting for application be read, then start jmeter'
sleep 60
kubectl --kubeconfig=remote-config apply -f ../jmeter_k8s.yaml
