build-sampling:
	docker build \
      --file=./Dockerfile-sampling.yaml \
      --tag=tuning/sampling ../

deploy-sampling:
	kubectl apply -f sampling-deployment.yml

delete-sampling:
	kubectl delete -f sampling-deployment.yml
