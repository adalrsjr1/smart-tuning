#!/bin/bash

kubectl get pod -A -o json | jq .items[].spec.containers[].resources?.limits?.cpu | grep -Eoh "[0-9]+" | awk  -F: '{if ( length($0) < 2 ) { print $0"000" } else { print }  }' | awk '{s+=$1} END {print "cpu limit: " s "m"}'
kubectl top pod --no-headers -A | awk '{ print $3 }' | grep -Eoh "[0-9]+" | awk '{s+=$1} END {print "cpu: " s "m"}'

echo ""
kubectl get pod -A -o json | jq .items[].spec.containers[].resources?.limits?.memory | grep -Eoh "[0-9]+" | awk  -F: '{if ( length($0) < 2 ) { print $0*1024 } else { print }  }' | awk '{s+=$1} END {print "memory limit: " s "m"}'
kubectl top pod --no-headers -A | awk '{ print $4 }' | grep -Eoh "[0-9]+" | awk '{s+=$1} END {print "memory: " s "Mi"}'
