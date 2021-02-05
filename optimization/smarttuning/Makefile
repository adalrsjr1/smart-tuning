TAG := $(shell git log -1 --format=%h)
.PHONY:
	build push

build-dev:
	@echo "building image: quay.io/smarttuning/smarttuning:dev and $(TAG)"
	docker build -t smarttuning:dev -t quay.io/smarttuning/smarttuning:dev -t smarttuning:$(TAG) -t quay.io/smarttuning/smarttuning:$(TAG) .

build:
	echo "building image: quay.io/smarttuning/smarttuning"
	docker build -t quay.io/smarttuning/smarttuning .
	docker build -t smarttuning .

push:
	#docker push quay.io/smarttuning/smarttuning
	docker push quay.io/smarttuning/smarttuning:dev

deploy:
	kubectl apply -f ../manifests/mongo-deployment.yaml -f ../manifests/tuning-deployment.yaml

delete:
	kubectl delete -f ../manifests/mongo-deployment.yaml -f ../manifests/tuning-deployment.yaml
