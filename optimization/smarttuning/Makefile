.PHONY:
	build push

build-dev:
	echo "building image: quay.io/smarttuning/smarttuning:dev"
	docker build -t quay.io/smarttuning/smarttuning:dev .
	docker build -t smarttuning:dev .

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
