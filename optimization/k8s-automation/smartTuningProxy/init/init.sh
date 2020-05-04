#!/bin/bash

# Forward TCP traffic on port 80 to port 8000 on the eth0 interface
iptables -t nat -A PREROUTING -p tcp -i eth0 --dport $SERVICE_PORT -j REDIRECT --to-port $PROXY_PORT

# List all iptables rules.
iptables -t nat --list
