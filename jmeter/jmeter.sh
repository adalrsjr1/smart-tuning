#!/bin/bash

apache-jmeter-5.2.1/bin/jmeter \
  -Dprometheus.ip=0.0.0.0 \
  -DusePureIDs=true \
  -JWAIT_RESP=3600
