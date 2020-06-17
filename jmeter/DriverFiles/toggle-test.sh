#!/bin/bash

E_NOARGS=85

if [[ $# -ne 3 ]]; then
  echo -e "usage: $0 <thread_group> <true|false> <test_plan.jmx>\n"
  echo -e "example: `basename $0` workload-1 true tuning.jmx\n"
  echo -e "options available:"
  echo -e "\tworkload-#, # between (0,5) including low and high"
  echo -e "\tworkload-#, # stands for (uniform, random, runtime)"
  exit $E_NOARGS
fi

TEST_GROUP="\"$1\""
OPTION=$2
TEST_PLAN=$3
REGEX='"[^"]*"'

if [[ $OPTION == "true" ]]; then
  echo "enabling $TEST_GROUP"
elif [[ $OPTION == "false" ]]; then
  echo "disabling $TEST_GROUP"
else
  echo "option $OPTION not valid, it should be <true|false>"
  exit $E_NOARGS
fi

OPTION="\"$OPTION\""
sed -i.bak "s/testname=$TEST_GROUP enabled=$REGEX/testname=$TEST_GROUP enabled=$OPTION/" $TEST_PLAN
rm *.bak 2> /dev/null

exit 0
