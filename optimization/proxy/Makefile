proxy:
	docker build -t smarttuning-proxy -f Dockerfile .
	docker build -t smarttuning/proxy -f Dockerfile .
	docker push smarttuning/proxy

iproxy:
	docker build -t smarttuning-init -f init/Dockerfile .
	docker build -t smarttuning/init-proxy -f init/Dockerfile .
	docker push smarttuning/init-proxy

deploy:
	kubectl apply -f manifests/proxy-cm.yaml -f manifests/proxy-test-dp.yaml

remove:
	kubectl delete -f manifests/proxy-cm.yaml -f manifests/proxy-test-dp.yaml

clean:
	docker rmi smarttuning-proxy
	docker rmi smarttuning/proxy
	docker rmi smarttuning-init
	docker rmi smarttuning/init-proxy

all: proxy iproxy
.PHONY: all
