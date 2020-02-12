#!/bin/bash
# The last parameter in the line below is the IP of the machine where liberty runs
#rm jmeter_output/raw_data.jtl
if [ -z "$1" ]; then
  echo "Usage: `basename $0` [workload-#] [# of repeats] [# of clients] [duration default=60s] [rampup default=0] [url default=localhost] [port default=80]"
  exit $E_NOARGS
fi

if [ -z "$2" ];then
  echo "Usage: `basename $0` [workload-#] [# of repeats] [# of clients] [duration default=60s] [rampup default=0] [url default=localhost] [port default=80]"
  exit $E_NOARGS
fi

if [ -z "$3" ];then
  echo "Usage: `basename $0` [workload-#] [# of repeats] [# of clients] [duration default=60s] [rampup default=0] [url default=localhost] [port default=80]"
  exit $E_NOARGS
fi

THREAD_GROUP=$1
TEST_PLAN=tuning-workloads.jmx

JREPEAT=$2
JCLIENTS=$3

if [ -z "$4" ];then
  JDURATION=60
else
  JDURATION=$4
fi

if [ -z "$5" ];then
  JRAMPUP=0
else
  JRAMPUP=$5
fi

if [ -z "$6" ]; then
  LIBERTYHOST='localhost'
else
  LIBERTYHOST=$6
fi

if [ -z "$7" ]; then
  JPORT=80
else
  JPORT=$7
fi

docker rm -f jmeter1 2>&1 >> /dev/null

echo "
RUN_JMETER: docker run \
-p 9270:9270 \
--cpuset-cpus='3' \
--rm --net=acmeair-network \
-v /Users/adalbertoibm.com/Coding/Dockerized_AcmeAir/jmeter_output:/output \
-e JTHREAD=$JCLIENTS \
-e JDURATION=$JDURATION \
-e LIBERTYHOST=$LIBERTYHOST \
-e JPORT=$JPORT \
-e JREPEAT=$JREPEAT \
-e JRAMPUP=$JRAMPUP \
-e THREAD_GROUP=$THREAD_GROUP \
-e TEST_PLAN=$TEST_PLAN \
--name jmeter1 \
jmeter_acmeair acmeair
"

docker run \
-p 9270:9270 \
--cpuset-cpus='3' \
--rm --net=acmeair-network \
-v /Users/adalbertoibm.com/Coding/Dockerized_AcmeAir/jmeter_output:/output \
-e JTHREAD=$JCLIENTS \
-e JDURATION=$JDURATION \
-e LIBERTYHOST=$LIBERTYHOST \
-e JPORT=$JPORT \
-e JREPEAT=$JREPEAT \
-e JRAMPUP=$JRAMPUP \
-e THREAD_GROUP=$THREAD_GROUP \
-e TEST_PLAN=$TEST_PLAN \
--name jmeter1 \
jmeter_acmeair acmeair
