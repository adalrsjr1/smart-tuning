#!/bin/bash

usage="$(basename "$0") [-h] [-a n] -- program to check if server is up

where:
    -h  show this help text
    -a  server address (optional) <address:port>"

URL="localhost"
while getopts ':ha:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    a) URL=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

while [[ true ]] ; do
  HEADERS=$(curl -Is $URL)
  HTTPSTATUS=$(echo $HEADERS | grep HTTP | cut -d' ' -f2)
  if [[ -z $HTTPSTATUS || $HTTPSTATUS -ge 399 ]]; then
    echo "waiting for endpoint $URL..."
    sleep 1
  else
    exit 0
  fi
done;
