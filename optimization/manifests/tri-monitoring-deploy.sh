#!/bin/bash
echo -e "deploying reloader\n"
tri apply -f  https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
sleep 1
echo -e "\ndeploying prometheus\n"
cat ../prometheus/* | tri apply -f  - --validate
sleep 1
echo -e "\ndeploying grafana\n"
cat ../grafana/grafana-dash.yaml | tri apply -f  -
cat ../grafana/grafana-deployment.yml | tri apply -f  -
sleep 1
echo -e "\ndeploying searchspace\n"
cat search-space/search-space-crd-2.yaml | tri apply -f -
sleep 1
echo -e "\ndeploying proxy config\n"
cat proxy-config.yaml | tri apply -f -
sleep 1
echo -e "\ndeploying mongo\n"
cat mongo-deployment.yaml | tri apply -f -
sleep 1
