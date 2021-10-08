#!/bin/bash

jmeter -n \
  -Dprometheus.ip=0.0.0.0
  -DusePureIDs=true \
  -t $JMETER_HOME/teastore.jmx \
  -JHOST=$JHOST \
  -JPORT=$JPORT \
  -JnumUsers=$JTHREADS \
  -JrampUp=$JRAMP \
  -JDURATION=$JDURATION \

