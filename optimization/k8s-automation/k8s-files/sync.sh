#!/bin/bash

rsync --ignore-existing -avvzu --delete  --exclude '.git' --exclude 'venv' --exclude '.venv' Dockerized_AcmeAir adalrsjr@trxrhel7perf-1.canlab.ibm.com:.
