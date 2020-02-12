#!/bin/bash

echo "python -u -m opt.optimization_acmeair\
    --prometheus-url $PROMETHEUS_URL\
    --prometheus-port $PROMETHEUS_PORT\
    --mongo-url $MONGO_URL\
    --mongo-port $MONGO_PORT\
    --mongo-db $DB_NAME\
    --mongo-collection $DB_COLLECTION \
    --application-url $APP_URL\
    --application-port $APP_PORT\
    --application-path $APP_PATH
"
echo 'starting...'
python -u -m opt.optimization_acmeair \
    --prometheus-url $PROMETHEUS_URL \
    --prometheus-port $PROMETHEUS_PORT \
    --mongo-url $MONGO_URL \
    --mongo-port $MONGO_PORT \
    --mongo-db $DB_NAME \
    --mongo-collection $DB_COLLECTION \
    --application-url $APP_URL \
    --application-port $APP_PORT \
    --application-path $APP_PATH \
    --n-iterations $N_ITERATIONS

