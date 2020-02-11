#!/bin/bash

echo "python acmeair_workload.py
    --prometheus-url $PROMETHEUS_URL
    --prometheus-port $PROMETHEUS_PORT
    --fetch_window $FETCH_WINDOW
    --time_unit $TIME_UNIT
    --mongo-url $MONGO_URL
    --mongo-port $MONGO_PORT
    --db-name $DB_NAME
    --db-collection $DB_COLLECTION
    --db-buffer-size $DB_BUFFER_SIZE
    --metric-name $METRIC_NAME
    --metric-query $METRIC_QUERY
"
python -u acmeair_workload.py\
    --prometheus-url $PROMETHEUS_URL\
    --prometheus-port $PROMETHEUS_PORT\
    --fetch_window $FETCH_WINDOW\
    --time_unit $TIME_UNIT\
    --mongo-url $MONGO_URL\
    --mongo-port $MONGO_PORT\
    --db-name $DB_NAME\
    --db-collection $DB_COLLECTION\
    --db-buffer-size $DB_BUFFER_SIZE\
    --metric-name $METRIC_NAME\
    --metric-query $METRIC_QUERY

