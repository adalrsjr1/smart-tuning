build:
	echo "building image: reloader"
	docker build -t reloader -f reloader/Dockerfile reloader/
	echo "building image: sync"
	docker build -t sync -f reloader/Dockerfile-sync reloader/
	echo "building image: smarttuning"
	docker build -t smarttuning -f optimization/Dockerfile optimization/

deploy:
	kubectl apply -f mongo-deployment.yaml -f tuning-deployment.yaml -f reload-deploymnet.yaml -f sync-deployment.yaml

delete:
	kubectl delete -f mongo-deployment.yaml -f tuning-deployment.yaml -f reload-deploymnet.yaml -f sync-deployment.yaml
