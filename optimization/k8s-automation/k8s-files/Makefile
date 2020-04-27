build:
	echo "building image: reloader"
	docker build -t reloader -f reloader/Dockerfile reloader/
	echo "building image: smarttuning"
	docker build -t smarttuning -f optimization/Dockerfile optimization/

deploy:
	kubectl apply -f mongo-deployment.yaml -f tuning-deployment.yaml -f reload-deploymnet.yaml

delete:
	kubectl delete -f mongo-deployment.yaml -f tuning-deployment.yaml -f reload-deploymnet.yaml