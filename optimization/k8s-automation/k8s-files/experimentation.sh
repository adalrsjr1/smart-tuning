#!/bin/bash

./deployment.sh
sleep 50
kubectl -f ../jmeter_k8s.yaml
sleep 14400
./delete.sh