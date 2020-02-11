#!/bin/bash

python3 -m opt.optimization_acmeair

echo "python -u -m opt.optimization_acmeair
    --prometheus-url $PROMETHEUS_URL
    --prometheus-port $PROMETHEUS_PORT
    --mongo-url $MONGO_URL
    --mongo-port $MONGO_PORT
    --mongo-db $DB_NAME
    --mongo-collection $DB_COLLECTION
"
echo 'starting...'
python -u -m opt.optimization_acmeair
    --prometheus-url $PROMETHEUS_URL
    --prometheus-port $PROMETHEUS_PORT
    --mongo-url $MONGO_URL
    --mongo-port $MONGO_PORT
    --mongo-db $DB_NAME
    --mongo-collection $DB_COLLECTION

