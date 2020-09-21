docker build -f Dockerfile_plain -t smarttuning/jmeter_plain .
docker push smarttuning/jmeter_acmeair

docker build -f Dockerfile_jmeter -t smarttuning/jmeter_acmeair .
docker push smarttuning/jmeter_acmeair
