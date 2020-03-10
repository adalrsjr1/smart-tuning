#!/bin/bash

echo "python acmeair_workflow.py
    --prometheus-url $PROMETHEUS_URL
    --prometheus-port $PROMETHEUS_PORT
    --fetch_window $FETCH_WINDOW
    --time_unit $TIME_UNIT
    --mongo-url $MONGO_URL
    --mongo-port $MONGO_PORT
    --db-name $DB_NAME
    --db-collection $DB_COLLECTION
    --db-buffer-size $DB_BUFFER_SIZE
"
echo 'starting...'
python -u acmeair_workflow.py\
    --prometheus-url $PROMETHEUS_URL\
    --prometheus-port $PROMETHEUS_PORT\
    --fetch_window $FETCH_WINDOW\
    --time_unit $TIME_UNIT\
    --mongo-url $MONGO_URL\
    --mongo-port $MONGO_PORT\
    --db-name $DB_NAME\
    --db-collection $DB_COLLECTION\
    --db-buffer-size $DB_BUFFER_SIZE

