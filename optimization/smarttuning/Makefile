build:
	echo "building image: smarttuning"
	docker build -t smarttuning .

deploy:
	kubectl apply -f ../manifests/mongo-deployment.yaml -f ../manifests/tuning-deployment.yaml

delete:
	kubectl delete -f ../manifests/mongo-deployment.yaml -f ../manifests/tuning-deployment.yaml
