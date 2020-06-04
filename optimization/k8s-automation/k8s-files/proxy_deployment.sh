#!/bin/bash

kubectl --kubeconfig remote-config apply -f acmeair-config.yaml
kubectl --kubeconfig remote-config apply -f acmeair-db-deployment.yaml
#
# no-proxy
kubectl --kubeconfig remote-config apply -f test_plain.yaml
sleep 60
kubectl --kubeconfig remote-config apply -f ../single_jmeter_k8s.yaml
sleep 1000
kubectl --kubeconfig remote-config delete -f ../single_jmeter_k8s.yaml
kubectl --kubeconfig remote-config delete -f test_plain.yaml
sleep 60 
## custom custom
kubectl --kubeconfig remote-config apply -f test_custom.yaml
sleep 60
kubectl --kubeconfig remote-config apply -f ../single_jmeter_k8s.yaml
sleep 1000
kubectl --kubeconfig remote-config delete -f ../single_jmeter_k8s.yaml
kubectl --kubeconfig remote-config delete -f test_custom.yaml
sleep 60
#
## custom envoy
kubectl --kubeconfig remote-config apply -f test_envoy.yaml
sleep 60
kubectl --kubeconfig remote-config apply -f ../single_jmeter_k8s.yaml
sleep 1000
kubectl --kubeconfig remote-config delete -f ../single_jmeter_k8s.yaml
kubectl --kubeconfig remote-config delete -f test_envoy.yaml
sleep 60
#
## custom traefik
kubectl --kubeconfig remote-config apply -f test_traefik.yaml
sleep 60
kubectl --kubeconfig remote-config apply -f ../single_jmeter_k8s.yaml
sleep 1000
kubectl --kubeconfig remote-config delete -f ../single_jmeter_k8s.yaml
kubectl --kubeconfig remote-config delete -f test_traefik.yaml
sleep 60
#
