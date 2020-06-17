# Manual for AcmeAir load test

## URLs

* `\login`
* `\view-profile`
* `\update-profile`
* `\query-flight`
* `\book-flight`*
* `\list-bookings`
* `\cancel-booking`*

\* their behavior changes over time since the database stretch and shrink
according to the execution of the application

## Load testing strategies

### Uniform distribution

All URLs, but `\book-flight` and `cancel-booking`, are sequentially accessed over
time.

### Random distribution

All URLs, but `\book-flight` and `cancel-booking`, are accessed over
time uniformly.

### Runtime distribution

JMeter simulates sessions durations between `MIN_SESSION_DURATION` and
`MAX_SESSION_DURATION`. During a session, all URLs are accessed as in random
distribution.

## Variables

* `LOAD_BOOKINGS`: load bookings into database, default `true`
* `USERS`: number of users signed up to the system, default `200`
* `URL`: context root, default `acmeair-webapp`
* `USER_BOTTOM`: low bounder of user id interval `(USER_BOTTOM, USER)`, default
  0
* `PORT`: port of host target, default `8080`
* `HOST`: host target, default `WLP_HOSTS`
* `WLP_HOSTS`: host target set in .csv file, default `localhost`
* `MIN_SESSION_DURATION`: min session duration while running Runtime distribution strategy,
  default `0`s
* `MAX_SESSION_DURATION`: max session duration while running Runtime distribution strategy,
default `60`s
* `MIN_THINK`: min thinking time when accessing each URL, default `100`ms
* `MAX_THINK`: max thinking time when accessing each URL, default `200`ms
* `THREAD`: number of threads/clients, default `10`
* `RAMPUP`: delay between each thread spawn time, default `0`
* `DURATION`: duration of load test, default `60`s
### Session duration

Based on this
[report](https://databox.com/average-session-duration-benchmark#difference)

### Thinking time

Gets a values in a Poisson distribution, lambda = `MIN_THINK` offset =
`MAX_THINK`. Using the
default values, the average of thinking time is between `275` and `3255`
