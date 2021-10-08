# Dockerfile to build a jmeter container able to drive acmeair
# Results appear on /output in the container
# Must specify the hostname for the acmeair application (or localhost will be assumed)
FROM smarttuning/jmeter_plain

# Copy the script to be executed and other needed files
COPY jmeter_files/toggle-test.sh $JMETER_HOME/bin/toggle-test.sh
COPY jmeter_files/dynamic_init.sh $JMETER_HOME/bin/init.sh
RUN chmod a+x $JMETER_HOME/bin/init.sh

# Environment variables that we want the user to redefine
ENV JHOST=localhost \
    JPORT=8080 \
    JTHREADS=10 \
    JRAMP=0 \
    JDURATION=60 \
    TEST_GROUP=single\
    TEST_PLAN=teastore.jmx

EXPOSE 9270

COPY jmeter_files/. $JMETER_HOME/

ENTRYPOINT ["/bin/bash",  "-c", "init.sh"]


