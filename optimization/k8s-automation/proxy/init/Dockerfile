FROM ubuntu

RUN apt-get update && \
    apt-get install -y iptables

COPY init/init.sh /usr/local/bin

ENV PROXY_PORT=8000 \
    SERVICE_PORT=80 

RUN chmod +x /usr/local/bin/init.sh

ENTRYPOINT ["init.sh"]
