FROM golang:alpine

RUN mkdir /proxy
WORKDIR /proxy

ENV GO111MODULE=on

COPY go.mod .
COPY go.sum .

RUN go mod download

COPY main.go main.go
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o /go/bin/proxy

ENTRYPOINT ["/go/bin/proxy"]
