#!/bin/bash

#GO111MODULE=on
#go mod download
#
#mkdir -p ./build
#CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -i -installsuffix cgo -o ./build/proxy
#
docker build -t smarttuning-proxy -f Dockerfile .
docker build -t smarttuning-init -f init/Dockerfile .

#rm -rf ./build
