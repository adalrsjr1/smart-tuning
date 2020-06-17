proxy:
	docker build -t smarttuning-proxy -f Dockerfile .

init:
	docker build -t smarttuning-init -f init/Dockerfile .

deploy:
	kubectl apply -f manifests/proxy-cm.yaml -f manifests/proxy-test-dp.yaml

remove:
	kubectl delete -f manifests/proxy-cm.yaml -f manifests/proxy-test-dp.yaml

all:
	proxy
	init
