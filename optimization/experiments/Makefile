.PHONY:
	build push

build:
	echo "building image: smarttuning"
	docker build -t smarttuning/replay .
	docker build -t replay .

push:
	docker push smarttuning/replay

