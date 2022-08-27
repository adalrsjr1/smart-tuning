#!/bin/bash

set -e

if [[ $# -ne 2 ]]; then
  echo -e "Usage: <namespace> <deployment>"
  exit 1
fi

NAMESPACE=$1
DEPLOYMENT=$2

#tee <<EOF /tmp/resources-patch-file.yaml
#spec:
#  template:
#    spec:
#      containers:
#        - name: $(echo "$NAME")
#          resources: {}
#EOF
#kubectl patch -n $NAMESPACE deployment $DEPLOYMENT --patch-file /tmp/resources-patch-file.yaml --type strategic

kubectl patch -n $NAMESPACE deployment $DEPLOYMENT --type=json -p='[{"op": "remove", "path": "/spec/template/spec/containers/0/resources/limits"}]'
