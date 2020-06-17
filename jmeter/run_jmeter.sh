#!/bin/bash

die() {
  printf '%s\n' "$1" >&2
  exit 1
}

usage() {
  echo "`basename $0` Usage:"
  echo -e "\t-h|--help|-\?\n\t\thelp\n"
  echo -e "\t-w|--workload\n\t\tworkload name\n"
  echo -e "\t-r|--repeats\n\t\tnumber of reptitions, default=0\n"
  echo -e "\t-c|--clients\n\t\tnumber of clients, default=100\n"
  echo -e "\t-d|--duration\n\t\tduration in secods of the test, default=60\n"
  echo -e "\t-u|--ramp-up\n\t\tramp-up duration, default=0\n"
  echo -e "\t-a|--acmeair-address\n\t\tacmeair address, default=localhost\n"
  echo -e "\t-p|--port\n\t\tacmeair address, default=9092\n"
  echo -e "\t-s|--min-session-time\n\t\tmin session time in seconds, default=0\n"
  echo -e "\t-S|--max-session-time\n\t\tmax session time in secnds, default=60\n"
  echo -e "\t-t|--min-think-time\n\t\tmin thinking time in miliseconds, default=100\n"
  echo -e "\t-T|--max-think-time\n\t\tmax thinking time in miliseconss, default=300\n"
}

LOAD_BOOKINGS=true
USERS=200
WORKLOAD=
REPEATS=1
USERS=200
CLIENTS=100
DURATION=60
RAMPUP=0
PORT=9092
ACMEAIR_ADDRESS=localhost
MIN_SESSION_DURATION=0  # seconds
MAX_SESSION_DURATION=60 # seconds
MIN_THINK=100   # milliseconds
MAX_THINK=200   # milliseconds
while :; do
  case $1 in
    -h|-\?|--help)
      usage
      exit
      ;;
    -w|--workload)
      if [ "$2" ]; then
        WORKLOAD=$2
        shift
      else
        die 'ERROR: "--workload requires a non-empty option argument.'
      fi
      ;;
    -r|--repeats)
      if [ "$2" ]; then
        REPEATS=$2
        shift
      else
        die 'ERROR: "--repeats requires a non-empty option argument.'
      fi
      ;;
    -c|--clients)
      if [ "$2" ]; then
        CLIENTS=$2
        shift
      else
        die 'ERROR: "--clients requires a non-empty option argument.'
      fi
      ;;
    -d|--duration)
      if [ "$2" ]; then
        DURATION=$2
        shift
      else
        die 'ERROR: "--duration requires a non-empty option argument.'
      fi
      ;;
    -u|--rump-up)
      if [ "$2" ]; then
        RUMPUP=$2
        shift
      else
        die 'ERROR: "--rump-up requires a non-empty option argument.'
      fi
      ;;
    -a|--acmeair-address)
      if [ "$2" ]; then
        ACMEAIR_ADDRESS=$2
        shift
      else
        die 'ERROR: "--acmeair-address requires a non-empty option argument.'
      fi
      ;;
    -p|--port)
      if [ "$2" ]; then
        shift
      else
        die 'ERROR: "--port requests a non-empty option argument.'
      fi
      ;;
    -s|--min-session-duration)
      if [ "$2" ]; then
        MIN_SESSION_DURATION=$2
        shift
      else
        die 'ERROR: "--min-session-duration requires a non-empty option argument.'
      fi
      ;;
    -S|--max-session-duration)
      if [ "$2" ]; then
        MAX_SESSION_DURATION=$2
        shift
      else
        die 'ERROR: "--max-session-duration requires a non-empty option argument.'
      fi
      ;;
    -t|--min-think-time)
      if [ "$2" ]; then
        MIN_THINK=$2
        shift
      else
        die 'ERROR: "--min-think-time requires a non-empty option argument.'
      fi
      ;;
    -T|--max-think-time)
      if [ "$2" ]; then
        MAX_THINK=$2
        shift
      else
        die 'ERROR: "--max-think-time requires a non-empty option argument.'
      fi
      ;;
    --) # End of all options.
      shift
      break
      ;;
    -?*)
      printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
      ;;
    *)  # Default case: No more options, so break out of the loop
      break
  esac
  shift
done

if [ -z "$WORKLOAD"  ]; then
  usage
  exit $E_NOARGS
fi

THREAD_GROUP=$WORKLOAD
TEST_PLAN=tuning-workloads.jmx

JREPEAT=$REPEATS
JTHREAD=$CLIENTS
JDURATION=$DURATION
JRAMPUP=$RAMPUP
LIBERTYHOST=$ACMEAIR_ADDRESS

JLOAD_BOOKINGS=$LOAD_BOOKINGS
JUSERS=$USERS
JMIN_SESSION_DURATION=$MIN_SESSION_DURATION
JMAX_SESSION_DURATION=$MAX_SESSION_DURATION
JMIN_THINK=$MIN_THINK
JMAX_THINK=$MAX_THINK

docker rm -f jmeter1 2>&1 >> /dev/null

mkdir -p jmeter_output

docker run \
-p 9270:9270 \
--cpuset-cpus='3' \
--rm --net=acmeair-network \
-v jmeter_output:/output \
-e JLOAD_BOOKINGS=$JLOAD_BOOKINGS \
-e JUSERS=$JUSERS \
-e LIBERTYHOST=$LIBERTYHOST \
-e JTHREAD=$JTHREAD \
-e JDURATION=$JDURATION \
-e JPORT=$JPORT \
-e JMIN_SESSION_DURATION=$JMIN_SESSION_DURATION \
-e JMAX_SESSION_DURATION=$JMAX_SESSION_DURATION \
-e JMIN_THINK=$JMIN_THINK \
-e JMAX_THINK=$JMAX_THINK \
-e JREPEAT=$JREPEAT \
-e JRAMPUP=$JRAMPUP \
-e THREAD_GROUP=$THREAD_GROUP \
-e TEST_PLAN=$TEST_PLAN \
--name jmeter1 \
jmeter_acmeair acmeair
