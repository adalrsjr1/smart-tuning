#!/bin/bash

echo -e "config: no_waiting; jmeter: no_waiting\n"
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f best-config_jmeter-no-waiting-response/
sleep 180
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f jmeter-manifests/workloads/single_dyn_jmeter_k8s_prod.yaml
sleep 1800

./delete.sh
sleep 180

echo -e "config: waiting; jmeter: no_waiting\n"
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f best-config_jmeter-waiting-response/
sleep 180
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f best-config_jmeter-no-waiting-response/01-jmeter-cm.yaml -f jmeter-manifests/workloads/single_dyn_jmeter_k8s_prod.yaml
sleep 1800

./delete.sh
sleep 180

echo -e "config: no_waiting; jmeter: waiting\n"
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f best-config_jmeter-no-waiting-response/
sleep 180
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f best-config_jmeter-waiting-response/01-jmeter-cm.yaml -f jmeter-manifests/workloads/single_dyn_jmeter_k8s_prod.yaml
sleep 1800

./delete.sh
sleep 180

echo -e "config: waiting; jmeter: waiting\n"
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f best-config_jmeter-waiting-response/
sleep 180
kubectl --kubeconfig=$HOME/.kube/trinity01/config apply -f jmeter-manifests/workloads/single_dyn_jmeter_k8s_prod.yaml
sleep 1800

./delete.sh
sleep 180
