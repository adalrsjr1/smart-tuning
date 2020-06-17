proxy:
	docker build -t smarttuning-proxy -f Dockerfile .

init:
	docker build -t smarttuning-init -f init/Dockerfile .

all:
	proxy
	init
