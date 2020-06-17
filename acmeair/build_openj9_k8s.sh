# Note: the IP of the mongo process is embedded into the image through mongo.properties file
docker build -f Dockerfile_openj9_acmeair_k8s -t liberty-acmeair-k8s .

