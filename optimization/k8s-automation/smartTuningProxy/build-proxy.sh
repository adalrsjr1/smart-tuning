#!/bin/bash
GO111MODULE=on
go mod download

mkdir -p ./build
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 sudo go build -i -installsuffix cgo -o ./build/proxy

docker build -t smarttuning-proxy:latest -f Dockerfile .
docker build -t smarttuning-init:latest -f init/Dockerfile .
