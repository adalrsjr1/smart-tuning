#!/bin/bash
if [[ "$#" -ne 1 ]]; then
  echo "args: missing path for --kubeconfig"
  exit 1
fi

echo -e "deploying reloader\n"
kubectl apply --kubeconfig=$1 -f  https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 1
echo -e "\deploying metrics-server"
kubectl apply --kubeconfig=$1 -f ../metrics-server
sleep 1
echo -e "\ndeploying prometheus\n"
kubectl apply --kubeconfig=$1 -f  ../prometheus
sleep 1
echo -e "\ndeploying grafana\n"
kubectl apply --kubeconfig=$1 -f  ../grafana
sleep 1
echo -e "\ndeploying searchspace\n"
kubectl apply --kubeconfig=$1 -f  search-space/search-space-crd-2.yaml
sleep 1
echo -e "\ndeploying proxy config\n"
kubectl apply --kubeconfig=$1 -f proxy-config.yaml
sleep 1
echo -e "\ndeploying mongo\n"
kubectl apply --kubeconfig=$1 -f  mongo-deployment.yaml
sleep 1
