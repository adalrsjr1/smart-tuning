#!/bin/bash

sudo sshfs -o allow_other,default_permissions aks@20.3.171.9:/home/aks ./home
