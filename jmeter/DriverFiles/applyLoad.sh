#!/bin/bash
cd $JMETER_HOME

if [ $# -gt 0 ]
  then
    sed -i 's/localhost/'$1'/g' hosts.csv
fi
FOLDER_NAME=$(date '+%Y%m%d-%H%M%S')
mkdir /output/$FOLDER_NAME
for i in $(seq 1 $JREPEAT); do

  DATE=$(date '+%Y%m%d%-H%M%S')

#rm /output/raw_data.jtl
#rm /output/report.csv

  touch /output/$FOLDER_NAME/raw_data_$DATE.jtl

  jmeter \
    -n -Dprometheus.ip=0.0.0.0 \
    -DusePureIDs=true \
    -t tuning-workloads.jmx \
    -l /output/$FOLDER_NAME/raw_data_$DATE.jtl \
    -j /output/$FOLDER_NAME/acmeair.stats.$DATE \
    -JRAMPUP=$JRAMPUP \
    -JPORT=$JPORT \
    -JUSERBOTTOM=$JUSERBOTTOM \
    -JUSER=$JUSER \
    -JURL=$JURL \
    -JTHREAD=$JTHREAD \
    -JDURATION=$JDURATION \
    -JWAIT=$JWAIT \
    -JLOAD_BOOKINGS=$JLOAD_BOOKINGS \
    -JUSERS=$JUSERS \
    -JMIN_SESSION_DURATION=$JMIN_SESSION_DURATION \
    -JMAX_SESSION_DURATION=$JMAX_SESSION_DURATION \
    -JTARGET_THROUGHPUT=$JTARGET_THROUGHPUT \
    -JMIN_THINK=$JMIN_THINK \
    -JMAX_THINK=$JMAX_THINK

  bin/./JMeterPluginsCMD.sh \
    --tool Reporter \
    --generate-csv /output/$FOLDER_NAME/report_$DATE.csv \
    --input-jtl /output/$FOLDER_NAME/raw_data_$DATE.jtl \
    --plugin-type AggregateReport

done
