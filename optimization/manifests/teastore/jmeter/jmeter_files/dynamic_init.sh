#!/bin/bash

toggle-test.sh $TEST_GROUP true $TEST_PLAN

jmeter -n \
  -Dprometheus.ip=0.0.0.0 \
  -DusePureIDs=true \
  -t $JMETER_HOME/$TEST_PLAN \
  -JTOPUID=$JTOPUID \
  -JHOST=$JHOST \
  -JPORT=$JPORT \
  -JnumUsers=$JTHREADS \
  -JRATIO=$JRATIO \
  -JTHROUGHPUT=$JTHROUGHPUT \
  -JSTEPTHREADS=$JSTEPTHREADS \
  -JrampUp=$JRAMP \
  -JMAXTHINKTIME=$JMAXTHINKTIME \
  -JSTOCKS=$JSTOCKS \
  -JDURATION=$JDURATION \
  -JQUOTES=$JQUOTES \
  -JSELLS=$JSELLS \
  -JBUYS=$JBUYS \
  -JWAIT_RESP=$JWAIT_RESP

