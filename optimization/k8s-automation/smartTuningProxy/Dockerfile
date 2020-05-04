# FROM golang:alpine as build-env
# # All these steps will be cached
# RUN mkdir /proxy
# WORKDIR /proxy
# 
# # Force go to use modules
# ENV GO111MODULE=on
# 
# # <- COPY go.mod and go.sum files to the workspace
# COPY go.mod .
# COPY go.sum .
# 
# # Get dependancies - will also be cached if we won't change mod/sum
# RUN go mod download
# 
# FROM build-env as builder
# # COPY the source code as the last step
# COPY . .
# 
# # Build the binary
# RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -i -installsuffix cgo -o /go/bin/proxy
# #FROM scratch 
FROM alpine

ENV PROXY_PORT=8888 \
    SERVICE_PORT=80 \
    METRICS_PORT=9090

# <- Second step to build minimal image
COPY build/proxy /go/bin/proxy
ENTRYPOINT ["/go/bin/proxy"]
