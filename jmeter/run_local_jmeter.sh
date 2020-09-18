#!/bin/bash

apache-jmeter-5.2.1/bin/jmeter \
  -Dprometheus.ip=0.0.0.0 \
  -DusePureIDs=true \
  -t DriverFiles/tuning-workloads.jmx \
  -JRAMPUP=0 \
  -JPORT=30088 \
  -JURL=acmeair-webapp \
  -JTHREAD=1 \
  -JDURATION=60 \
  -JWAIT=0

