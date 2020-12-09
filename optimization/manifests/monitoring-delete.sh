#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "args: missing --kubeconfig file"
  exit 1
fi
echo -e "deploying reloader\n"
kubectl delete --kubeconfig=$1 -f  https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 1
echo -e "\ndeploying prometheus\n"
kubectl delete --kubeconfig=$1 -f  ../prometheus
sleep 1
echo -e "\ndeploying grafana\n"
kubectl delete --kubeconfig=$1 -f  ../grafana
sleep 1
echo -e "\ndeploying searchspace\n"
kubectl delete --kubeconfig=$1 -f  search-space/search-space-crd-2.yaml
sleep 1
echo -e "\nproxy config\n"
kubectl delete --kubeconfig=$1.yaml
sleep 1
kubectl delete --kubeconfig=$1 -f  mongo-deployment.yaml
sleep 1
