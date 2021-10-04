## General Config

* **AcmeAir**: <github.com/adalrsjr1/acmeair-monolithic-java>
* **Daytrader**: <github.com/adalrsjr1/sample.daytrader8>
* **Quarkus**: <github.com/adalrsjr1/quarkusRestCrudDemo>

**Don't refer to the original repositories for the following applications, since
their configuration files had to be customized to use parameters specified as
`envvar`.

Application | iteration length | workload types**
------------|------------------|---------------
AcmeAir     | 10 min           | 50, 100, 200
Daytrader   | 20 min           | 5, 10, 50 or JSP, JSF
Quarkus     | 5 min            | 50, 100, 200

** refer to `jmeter/workloads/[driver]/02-driver-job.yaml`

### AcmeAir

No specific action needed

### Daytrader

Daytrader need that its database have a limited size. Run the following scripts
inside the db container. To log into the container execute:

```kubectl exec -it db-container-name -n namespace -- bash```

#### Checking DB size

For db2 deployment run the following script as db2inst1 user: `su db2inst1`

```bash
#!/bin/bash

db2 connect to tradedb

while true; do db2 "select * from
  (select count (*) ACCOUNTEJB from ACCOUNTEJB) as ACCOUNTEJB,
  (select count (*) ACCOUNTPROFILEEJB from ACCOUNTPROFILEEJB) as ACCOUNTPROFILEEJB,
  (select count (*) HOLDINGEJB from HOLDINGEJB) as HOLDINGEJB,
  (select count (*) KEYGENEJB from KEYGENEJB) as KEYGENEJB,
  (select count (*) ORDEREJB from ORDEREJB) as ORDEREJB,
  (select count (*) QUOTEEJB from QUOTEEJB) as QUOTEEJB" ; sleep 1 ; done
```

#### Limiting DB size

For db2 deployment run the following script as db2inst1 user: `su db2inst1`.
Recommend cap tables size to 5000 rows.

```bash
#!/bin/bash

db2 connect to tradedb

while true ; do
  db2 "DELETE FROM (SELECT * FROM HOLDINGEJB FETCH FIRST $(( $(db2 connect to tradedb 2>&1 > /dev/null ; db2 "select count (*) HOLDINGEJB from HOLDINGEJB" | grep -E "[0-9]+" | tail -n 2 | head -n 1) - 2596)) ROWS ONLY) AS A";
  db2 "DELETE FROM (SELECT * FROM ORDEREJB FETCH FIRST $(( $(db2 connect to tradedb 2>&1 > /dev/null ; db2 "select count (*) ORDEREJB from ORDEREJB" | grep -E "[0-9]+" | tail -n 2 | head -n 1) - 2596)) ROWS ONLY) AS A";

  sleep 1; done;
```

Let the scrip running in background `./scrip.sh > /dev/null 2>&1 &`

### Quarkus HTTP crud

No specific action needed

## Useful commands

<https://kubernetes.io/docs/reference/kubectl/cheatsheet/>

#### Accessing application in K8s as localhost


`kubectl port-forward  service/daytrader -n daytrader 9080`

#### Watching logs of application in K8s

`kubectl logs -f pod-name -n namespace`
