package main

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"log"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/valyala/fasthttp"
)

var (
	// create helper function to get env var or fallback
	metricID      = getEnvOrDefault("METRIC_ID", "smarttuning")
	asyncProxy, _ = strconv.ParseBool(getEnvOrDefault("ASYNC", "false"))

	proxyPort, _   = strconv.Atoi(getEnvOrDefault("PROXY_PORT", "80"))
	metricsPort, _ = strconv.Atoi(getEnvOrDefault("METRICS_PORT", "9090"))
	upstreamAddr   = "127.0.0.1:" + getEnvOrDefault("SERVICE_PORT", "8080")

	// https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/
	podName           = getEnvOrDefault("POD_NAME", "")
	nodeName          = getEnvOrDefault("NODE_NAME", "")
	podNamespace      = getEnvOrDefault("POD_NAMESPACE", "default")
	svcName           = getEnvOrDefault("HOST_IP", "")
	podIP             = getEnvOrDefault("POD_IP", "")
	podServiceAccount = getEnvOrDefault("POD_SERVICE_ACCOUNT", "")

	training = getEnvOrDefault("TRAINING", "false")

	maxConn, _      = strconv.Atoi(getEnvOrDefault("MAX_CONNECTIONS", "10000"))
	readBuffer, _   = strconv.Atoi(getEnvOrDefault("READ_BUFFER_SIZE", "4096"))
	writeBuffer, _  = strconv.Atoi(getEnvOrDefault("WRITE_BUFFER_SIZE", "4096"))
	readTimeout, _  = strconv.Atoi(getEnvOrDefault("READ_TIMEOUT", "30"))
	writeTimeout, _ = strconv.Atoi(getEnvOrDefault("WRITE_TIMEOUT", "30"))
	connDuration, _ = strconv.Atoi(getEnvOrDefault("MAX_IDLE_CONNECTION_DURATION", "60"))
	connTimeout, _  = strconv.Atoi(getEnvOrDefault("MAX_CONNECTION_TIMEOUT", "30"))

	proxyClient = &fasthttp.HostClient{
		Addr:                          upstreamAddr,
		NoDefaultUserAgentHeader:      true, // Don't send: User-Agent: fasthttp
		MaxConns:                      maxConn,
		ReadBufferSize:                readBuffer,  // Make sure to set this big enough that your whole request can be read at once.
		WriteBufferSize:               writeBuffer, // Same but for your response.
		ReadTimeout:                   time.Second * time.Duration(readTimeout),
		WriteTimeout:                  time.Second * time.Duration(writeTimeout),
		MaxIdleConnDuration:           time.Second * time.Duration(connDuration),
		MaxConnWaitTimeout:            time.Second * time.Duration(connTimeout),
		DisableHeaderNamesNormalizing: true, // If you set the case on your headers correctly you can enable this.
	}

	countingRequests, _          = strconv.ParseBool(getEnvOrDefault("COUNT_REQUESTS", "true"))
	countingProcessTime, _       = strconv.ParseBool(getEnvOrDefault("COUNT_PROC_TIME", "true"))
	countingReqSize, _           = strconv.ParseBool(getEnvOrDefault("COUNT_REQ_SIZE", "true"))
	countingInRequests, _        = strconv.ParseBool(getEnvOrDefault("COUNT_IN_REQ", "true"))
	countingOutRequests, _       = strconv.ParseBool(getEnvOrDefault("COUNT_OUT_REQ", "true"))
	countingActiveConnections, _ = strconv.ParseBool(getEnvOrDefault("COUNT_ACTIVE_CONNS", "true"))
	instrumenting, _             = strconv.ParseBool(getEnvOrDefault("INSTRUMENTING", training))
	pathSeparator                = getEnvOrDefault("PATH_SEPARATOR", "&")
	pathSeparatorIndex, _        = strconv.Atoi(getEnvOrDefault("PATH_SEPARATOR_INDEX", "0"))

	httpRequestsTotal   *prometheus.CounterVec
	httpProcessTimeHist *prometheus.SummaryVec
	activeConnections   *prometheus.CounterVec
	httpSize            *prometheus.CounterVec
	inTotal             *prometheus.CounterVec
	outTotal            *prometheus.CounterVec

	reqChan = make(chan fasthttp.RequestCtx, maxConn)
)

func initPromCounters() {
	labels := []string{"training", "node", "pod", "namespace", "code", "path", "src", "dst"}

	if countingRequests {
		log.Println("Creating requests counter")
		httpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
			// when change this name also update prometheus config file
			Name: metricID + "_http_requests_total",
			Help: "Count of all HTTP requests",
		}, labels)
	}

	if countingProcessTime {
		log.Println("Creating process time counter")
		httpProcessTimeHist = promauto.NewSummaryVec(prometheus.SummaryOpts{
			Name:       metricID + "_http_processtime_seconds",
			Help:       "process time",
			Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001, 1.00: 0.00},
		}, labels)
	}

	if countingReqSize {
		log.Println("Creating request size counter")
		httpSize = promauto.NewCounterVec(prometheus.CounterOpts{
			Name: metricID + "_http_size",
			Help: "Traffic between nodes",
		}, append(labels, "direction"))
	}

	if countingInRequests {
		log.Println("Creating in_requests counter")
		inTotal = promauto.NewCounterVec(prometheus.CounterOpts{
			// when change this name also update prometheus config file
			Name: metricID + "_in_http_requests_total",
			Help: "Count of all HTTP requests",
		}, labels)
	}

	if countingOutRequests {
		log.Println("Creating out_requests counter")
		outTotal = promauto.NewCounterVec(prometheus.CounterOpts{
			// when change this name also update prometheus config file
			Name: metricID + "_out_http_requests_total",
			Help: "Count of all HTTP responses",
		}, labels)
	}

	if countingActiveConnections {
		log.Println("Creating active_conns gauge")
		activeConnections = promauto.NewCounterVec(prometheus.CounterOpts{
			Name: metricID + "_active_conns",
			Help: "Count the number of current active connections",
		}, []string{"training", "node", "pod", "namespace", "state"})
	}
}

func getEnvOrDefault(key string, fallback string) string {
	val, unset := os.LookupEnv(key)
	if !unset {
		val = fallback
	}
	log.Printf("%s=%s", key, val)
	return val
}

type PromMetric struct {
	path         []byte
	statusCode   int
	startTime    time.Time
	endTime      time.Duration
	requestsSize int
	responseSize int
	client       net.IP
	podIP        string
}

func SyncReverseProxyHandler(ctx *fasthttp.RequestCtx) {
	// TOOD: make this async
	req := &ctx.Request
	resp := &ctx.Response
	client := ctx.RemoteIP()
	tStart := ctx.ConnTime()

	requestSize := len(req.Body()) + len(req.URI().QueryString())

	prepareRequest(req, ctx)
	if err := proxyClient.Do(req, resp); err != nil {
		ctx.Logger().Printf("[%d -> %d] %s: %s", resp.StatusCode(), fasthttp.StatusBadGateway, string(req.RequestURI()), err)
		resp.SetStatusCode(fasthttp.StatusBadGateway)
	}

	responseSize := resp.Header.ContentLength()
	postprocessResponse(resp, ctx)
	tEnd := time.Since(tStart)

	// fix inconsistent URL with channels
	promMetric := PromMetric{
		path:         req.RequestURI(),
		statusCode:   resp.StatusCode(),
		startTime:    tStart,
		endTime:      tEnd,
		requestsSize: requestSize,
		responseSize: responseSize,
		client:       client,
		podIP:        podIP,
	}

	defer func(metric PromMetric) {
		// TODO: potential bug
		if metric.responseSize < 0 {
			metric.responseSize = 0
		}

		strPath := string(metric.path)
		strPath = strings.Split(strPath, pathSeparator)[pathSeparatorIndex]

		code := strconv.Itoa(metric.statusCode)
		if httpRequestsTotal != nil {
			httpRequestsTotal.With(prometheus.Labels{
				"training":  training,
				"node":      nodeName,
				"pod":       podName,
				"namespace": podNamespace,
				"code":      code,
				"path":      strPath, //+ p.sanitizeURLQuery(req.URL.RawQuery)
				"src":       metric.client.String(),
				"dst":       metric.podIP,
			}).Inc()
		}

		if httpProcessTimeHist != nil {
			httpProcessTimeHist.With(prometheus.Labels{
				"training":  training,
				"node":      nodeName,
				"pod":       podName,
				"namespace": podNamespace,
				"path":      strPath,
				"code":      code,
				"src":       metric.client.String(),
				"dst":       metric.podIP,
			}).Observe(metric.endTime.Seconds())
		}

		if httpSize != nil {
			// src -> dst
			httpSize.With(prometheus.Labels{
				"training":  training,
				"direction": "forward",
				"node":      nodeName,
				"pod":       podName,
				"namespace": podNamespace,
				"code":      code,
				"path":      strPath,
				//"size":      strconv.Itoa(metric.requestsSize),
				"src": metric.client.String(),
				"dst": metric.podIP,
			}).Add(float64(metric.requestsSize))

			// dst -> src
			httpSize.With(prometheus.Labels{
				"training":  training,
				"direction": "backward",
				"node":      nodeName,
				"pod":       podName,
				"namespace": podNamespace,
				"code":      code,
				"path":      strPath,
				//"size":      strconv.Itoa(metric.responseSize),
				"src": metric.podIP,
				"dst": metric.client.String(),
			}).Add(float64(metric.responseSize))
		}
	}(promMetric)
}

func AsyncReverseProxyHandler(ctx *fasthttp.RequestCtx) {
	// TOOD: make this async
	req := &ctx.Request
	//resp := &ctx.Response
	defer func() {
		ctx.Response.SetStatusCode(fasthttp.StatusOK)
	}()

	otherCtx := &fasthttp.RequestCtx{}
	otherCtx.Init(req, ctx.RemoteAddr(), ctx.Logger())

	go func(ctx *fasthttp.RequestCtx) {
		client := ctx.RemoteIP()
		tStart := ctx.ConnTime()

		requestSize := len(req.Body()) + len(req.URI().QueryString())

		prepareRequest(req, ctx)
		resp := fasthttp.AcquireResponse()
		if err := proxyClient.Do(req, resp); err != nil {
			ctx.Logger().Printf("[%d -> %d] %s: %s", resp.StatusCode(), fasthttp.StatusBadGateway, string(req.RequestURI()), err)
			resp.SetStatusCode(fasthttp.StatusBadGateway)
		}

		responseSize := resp.Header.ContentLength()
		postprocessResponse(resp, ctx)
		tEnd := time.Since(tStart)

		// fix inconsistent URL with channels
		promMetric := PromMetric{
			path:         req.RequestURI(),
			statusCode:   resp.StatusCode(),
			startTime:    tStart,
			endTime:      tEnd,
			requestsSize: requestSize,
			responseSize: responseSize,
			client:       client,
			podIP:        podIP,
		}

		defer func(metric PromMetric) {
			// TODO: potential bug
			if metric.responseSize < 0 {
				metric.responseSize = 0
			}

			strPath := string(metric.path)
			strPath = strings.Split(strPath, pathSeparator)[pathSeparatorIndex]

			code := strconv.Itoa(metric.statusCode)
			if httpRequestsTotal != nil {
				httpRequestsTotal.With(prometheus.Labels{
					"training":  training,
					"node":      nodeName,
					"pod":       podName,
					"namespace": podNamespace,
					"code":      code,
					"path":      strPath, //+ p.sanitizeURLQuery(req.URL.RawQuery)
					"src":       metric.client.String(),
					"dst":       metric.podIP,
				}).Inc()
			}

			if httpProcessTimeHist != nil {
				httpProcessTimeHist.With(prometheus.Labels{
					"training":  training,
					"node":      nodeName,
					"pod":       podName,
					"namespace": podNamespace,
					"path":      strPath,
					"code":      code,
					"src":       metric.client.String(),
					"dst":       metric.podIP,
				}).Observe(metric.endTime.Seconds())
			}

			if httpSize != nil {
				// src -> dst
				httpSize.With(prometheus.Labels{
					"training":  training,
					"direction": "forward",
					"node":      nodeName,
					"pod":       podName,
					"namespace": podNamespace,
					"code":      code,
					"path":      strPath,
					//"size":      strconv.Itoa(metric.requestsSize),
					"src": metric.client.String(),
					"dst": metric.podIP,
				}).Add(float64(metric.requestsSize))

				// dst -> src
				httpSize.With(prometheus.Labels{
					"training":  training,
					"direction": "backward",
					"node":      nodeName,
					"pod":       podName,
					"namespace": podNamespace,
					"code":      code,
					"path":      strPath,
					//"size":      strconv.Itoa(metric.responseSize),
					"src": metric.podIP,
					"dst": metric.client.String(),
				}).Add(float64(metric.responseSize))
			}
		}(promMetric)
	}(otherCtx)
}

func prepareRequest(req *fasthttp.Request, ctx *fasthttp.RequestCtx) {
	// do not proxy "Connection" header.
	ctxReq := &ctx.Request
	req.SetHost(proxyClient.Addr)

	clientIp := ctx.RemoteIP().String()

	xff := string(req.Header.Peek("X-Forwared-For"))
	if len(xff) <= 0 {
		req.Header.Set("X-Forwarded-For", clientIp)
	} else {
		req.Header.Set("X-Forwarded-For", xff+", "+clientIp)
	}

	req.Header.Set("X-Forwarded-Host", proxyClient.Addr)

	strPath := string(ctxReq.RequestURI())
	strPath = strings.Split(strPath, pathSeparator)[pathSeparatorIndex]

	if inTotal != nil {
		inTotal.With(prometheus.Labels{
			"training":  training,
			"node":      nodeName,
			"pod":       podName,
			"namespace": podNamespace,
			"code":      "0",
			"path":      strPath, //+ p.sanitizeURLQuery(req.URL.RawQuery)
			"src":       clientIp,
			"dst":       podIP,
		}).Inc()
	}
	//req.Header.Del("Connection")
	// strip other unneeded headers.

	// alter other request params before sending them to upstream host

}

func postprocessResponse(resp *fasthttp.Response, ctx *fasthttp.RequestCtx) {
	// do not proxy "Connection" header
	//resp.Header.Del("Connection")

	lastPod := string(ctx.Request.Header.Peek("X-Forwared-For"))
	if len(lastPod) <= 0 {
		resp.Header.Set("Pod", podName)
	} else {
		resp.Header.Set("Pod", lastPod+", "+podName)
	}

	ctxReq := &ctx.Request
	strPath := string(ctxReq.RequestURI())
	strPath = strings.Split(strPath, pathSeparator)[pathSeparatorIndex]
	if outTotal != nil {
		outTotal.With(prometheus.Labels{
			"training":  training,
			"node":      nodeName,
			"pod":       podName,
			"namespace": podNamespace,
			"code":      strconv.Itoa(resp.StatusCode()),
			"path":      strPath, //+ p.sanitizeURLQuery(req.URL.RawQuery)
			"src":       ctx.RemoteIP().String(),
			"dst":       podIP,
		}).Inc()
	}

	// strip other unneeded headers

	// alter other response data if needed

}

func main() {
	initPromCounters()
	go func() {
		r := http.NewServeMux()
		r.Handle("/metrics", promhttp.Handler())

		if instrumenting {
			r.HandleFunc("/debug/pprof/", pprof.Index)
			r.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
			r.HandleFunc("/debug/pprof/profile", pprof.Profile)
			r.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
			r.HandleFunc("/debug/pprof/trace", pprof.Trace)
		}

		if err := http.ListenAndServe(fmt.Sprintf(":%d", metricsPort), r); err != nil {
			log.Fatalf("error in pprof server: %s", err)
		}
	}()

	var handler func(ctx *fasthttp.RequestCtx)

	log.Println("Async Proxy =", asyncProxy)
	if asyncProxy {
		handler = AsyncReverseProxyHandler
	} else {
		handler = SyncReverseProxyHandler
	}

	s := &fasthttp.Server{
		Handler:      handler,
		ReadTimeout:  time.Second * time.Duration(readTimeout),
		WriteTimeout: time.Second * time.Duration(writeTimeout),
		IdleTimeout:  time.Second * time.Duration(connDuration),
		// couting active connections when Server Conn change its state
		ConnState: func(conn net.Conn, state fasthttp.ConnState) {
			if activeConnections != nil {
				activeConnections.With(prometheus.Labels{
					"training":  training,
					"node":      nodeName,
					"pod":       podName,
					"namespace": podNamespace,
					"state":     state.String(),
				}).Inc()
			}
		},
	}

	if err := s.ListenAndServe(fmt.Sprintf(":%d", proxyPort)); err != nil {
		log.Fatalf("error in fasthttp server: %s", err)
	}
}
