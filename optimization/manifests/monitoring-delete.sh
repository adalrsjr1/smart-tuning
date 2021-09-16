#!/bin/bash
CFG=$HOME/.kube/config
if [[ "$#" -ne 1 ]]; then
  echo "removing at localhost"
  #echo "args: missing path for --kubeconfig"
  #exit 1
else
  CFG=$1
fi


echo -e "removing reloader\n"
kubectl delete --kubeconfig=$CFG -f  https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 1
echo -e "\nremoving metrics-server\n"
kubectl delete --kubeconfig=$CFG -f ../metrics-server
echo -e "\nremoving kube state metrics\n"
kubectl delete --kubeconfig=$CFG -f ../kube-state-metrics
sleep 1
echo -e "\nremoving prometheus\n"
kubectl delete --kubeconfig=$CFG -f  ../prometheus
sleep 1
echo -e "\nremoving grafana\n"
kubectl delete --kubeconfig=$CFG -f  ../grafana
sleep 1
echo -e "\nremoving searchspace\n"
kubectl delete --kubeconfig=$CFG -f  search-space/search-space-crd-2.yaml
sleep 1
echo -e "\nremoving proxy config\n"
kubectl delete --kubeconfig=$CFG -f proxy-config.yaml
sleep 1
echo -e "\nremoving mongo\n"
kubectl delete --kubeconfig=$CFG -f  mongo-deployment.yaml
sleep 1
echo -e "\nremoving netutil pod\n"
kubectl --kubeconfig=$CFG delete pod netutil
echo -e "\nremoving dashboard\n"
kubectl delete --kubeconfig=$CFG -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0/aio/deploy/recommended.yaml
echo -e "to access dashboard: "
echo -e "https://github.com/kubernetes/dashboard/blob/master/docs/user/access-control/creating-sample-user.md"
