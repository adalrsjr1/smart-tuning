#!/bin/bash
echo -e "deploying reloader\n"
tri delete -f  https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 1
echo -e "\ndeploying prometheus\n"
cat ../prometheus/* | tri delete -f -
sleep 1
echo -e "\ndeploying grafana\n"
cat ../grafana/*.y* | tri delete -f -
sleep 1
echo -e "\ndeploying searchspace\n"
cat search-space/search-space-crd-2.yaml | tri delete -f -
sleep 1
echo -e "\nproxy config\n"
cat proxy-config.yaml | tri delete -f -
sleep 1
cat mongo-deployment.yaml | tri delete -f -
sleep 1
