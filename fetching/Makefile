test:
	python -m unittest tests/test_fetcher.py

docker-build-workflow:
	docker build \
	  --file=./Dockerfile_workflow.yaml \
      --tag=acmeair/hist_fetcher ../

docker-build-workload:
	docker build \
      --file=./Dockerfile_workload.yaml \
      --tag=acmeair/metric_fetcher ../

docker-build: docker-build-workflow docker-build-workload

docker-run-workflow:
	docker run -it --rm --net=host acmeair/hist_fetcher

docker-run-workload:
	docker run -it --rm --net=host acmeair/metric_fetcher
