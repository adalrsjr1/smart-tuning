test:

docker-build-optimization:
	docker build \
	  --file=./Dockerfile_optimization.yaml \
      --tag=acmeair/optimization ../

docker-build-worker:
	docker build \
      --file=./Dockerfile_worker \
      --tag=acmeair/opt-worker ../

docker-build: docker-build-optimization

docker-run-optimization:
	docker run -it --rm --net=host -v /var/run/docker.sock:/var/run/docker.sock acmeair/optimization

docker-run-worker:
	docker run -it --rm --net=host acmeair/worker
