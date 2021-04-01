#!/bin/bash
CFG=$HOME/.kube/config
if [[ "$#" -ne 1 ]]; then
  echo "deploying at localhost"
  #echo "args: missing path for --kubeconfig"
  #exit 1
else
  CFG=$1
fi

echo -e "deploying reloader\n"
kubectl delete --kubeconfig=$CFG -f  https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 1
echo -e "\ndeploying metrics-server\n"
kubectl delete --kubeconfig=$CFG -f ../metrics-server
echo -e "\ndeploying kube state metrics\n"
kubectl delete --kubeconfig=$CFG -f ../kube-state-metrics
sleep 1
sleep 1
echo -e "\ndeploying prometheus\n"
kubectl delete --kubeconfig=$CFG -f  ../prometheus
sleep 1
echo -e "\ndeploying grafana\n"
kubectl delete --kubeconfig=$CFG -f  ../grafana
sleep 1
echo -e "\ndeploying searchspace\n"
kubectl delete --kubeconfig=$CFG -f  search-space/search-space-crd-2.yaml
sleep 1
echo -e "\ndeploying proxy config\n"
kubectl delete --kubeconfig=$CFG -f proxy-config.yaml
sleep 1
echo -e "\ndeploying mongo\n"
kubectl delete --kubeconfig=$CFG -f  mongo-deployment.yaml
sleep 1
echo -e "\ndeploying netutil pod\n"
kubectl --kubeconfig=$CFG run netutil --image=amouat/network-utils --image-pull-policy=IfNotPresent --command -- "/bin/sh" "-c" "while true; do sleep 10; done;"
echo -e "\ndeploying dashboard\n"
kubectl delete --kubeconfig=$CFG -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0/aio/deploy/recommended.yaml
echo -e "to access dashboard: "
echo -e "https://github.com/kubernetes/dashboard/blob/master/docs/user/access-control/creating-sample-user.md"

