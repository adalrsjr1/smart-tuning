#!/bin/bash

echo "toggle-test $THREAD_GROUP true $JMETER_HOME/$TEST_PLAN"
toggle-test.sh $THREAD_GROUP true $JMETER_HOME/$TEST_PLAN

echo $RETVAL
if [[ $RETVAL -ne 0 ]]; then
  echo "ERROR to toggle on $THREAD_GROUP"
  exit 1
fi

applyLoad.sh $LIBERTYHOST
