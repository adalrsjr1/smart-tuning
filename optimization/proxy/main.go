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
	metricID            = getEnvOrDefault("METRIC_ID", "smarttuning")

	proxyPort, _    = strconv.Atoi(getEnvOrDefault("PROXY_PORT", "80"))
	metricsPort, _  = strconv.Atoi(getEnvOrDefault("METRICS_PORT", "9090"))
	upstreamAddr    = "127.0.0.1:" + getEnvOrDefault("SERVICE_PORT", "8080")

	// https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/
	podName           = getEnvOrDefault("POD_NAME", "")
	nodeName          = getEnvOrDefault("NODE_NAME", "")
	podNamespace      = getEnvOrDefault("POD_NAMESPACE", "default")
	svcName           = getEnvOrDefault("HOST_IP", "")
	podIP             = getEnvOrDefault("POD_IP", "")
	podServiceAccount = getEnvOrDefault("POD_SERVICE_ACCOUNT", "")

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
		ReadTimeout:                   time.Duration(readTimeout) * time.Second,
		WriteTimeout:                  time.Duration(writeTimeout) * time.Second,
		MaxIdleConnDuration:           time.Duration(connDuration) * time.Second,
		MaxConnWaitTimeout:            time.Duration(connTimeout) * time.Second,
		DisableHeaderNamesNormalizing: true, // If you set the case on your headers correctly you can enable this.
	}

	countingRequests, _     = strconv.ParseBool(getEnvOrDefault("COUNT_REQUESTS", "true"))
	countingProcessTime, _  = strconv.ParseBool(getEnvOrDefault("COUNT_PROC_TIME", "true"))
	countingReqSize, _		= strconv.ParseBool(getEnvOrDefault("COUNT_REQ_SIZE", "true"))
	countingInRequests, _	= strconv.ParseBool(getEnvOrDefault("COUNT_IN_REQ", "true"))
	countingOutRequests, _	= strconv.ParseBool(getEnvOrDefault("COUNT_OUT_REQ", "true"))
	instrumenting, _ 		= strconv.ParseBool(getEnvOrDefault("INSTRUMENTING", "false"))

	httpRequestsTotal 		*prometheus.CounterVec
	httpProcessTimeHist 	*prometheus.SummaryVec
	httpSize 				*prometheus.CounterVec
	inTotal 				*prometheus.CounterVec
	outTotal 				*prometheus.CounterVec

	promChan = make(chan PromMetric, maxConn)
)

func initPromCounters() {
	if countingRequests {
		log.Println("Creating requests counter")
		httpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
			// when change this name also update prometheus config file
			Name: metricID + "_http_requests_total",
			Help: "Count of all HTTP requests",
		}, []string{"node", "pod", "namespace", "code", "path", "src", "dst"})
	}

	if countingProcessTime {
		log.Println("Creating process time counter")
		httpProcessTimeHist = promauto.NewSummaryVec(prometheus.SummaryOpts{
			Name:       metricID + "_http_processtime_seconds",
			Help:       "process time",
			Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001, 1.00: 0.00},
		}, []string{"node", "pod", "namespace", "code", "path", "src", "dst"})
	}

	if countingReqSize {
		log.Println("Creating request size counter")
		httpSize = promauto.NewCounterVec(prometheus.CounterOpts{
			Name: metricID + "_http_size",
			Help: "Traffic between nodes",
		}, []string{"direction", "node", "pod", "namespace", "code", "path", "src", "dst"})
	}

	if countingInRequests {
		log.Println("Creating in_requests counter")
		inTotal = promauto.NewCounterVec(prometheus.CounterOpts{
			// when change this name also update prometheus config file
			Name: "in_http_requests_total",
			Help: "Count of all HTTP requests",
		}, []string{"node", "pod", "namespace", "code", "path", "src", "dst"})
	}

	if countingOutRequests {
		log.Println("Creating out_requests counter")
		outTotal = promauto.NewCounterVec(prometheus.CounterOpts{
			// when change this name also update prometheus config file
			Name: "out_http_requests_total",
			Help: "Count of all HTTP responses",
		}, []string{"node", "pod", "namespace", "code", "path", "src", "dst"})
	}
}

func getEnvOrDefault(key string, fallback string) string {
	val, unset := os.LookupEnv(key)

	if !unset {
		val = fallback
	}
	return val
}

type PromMetric struct {
	path         []byte
	statusCode   int
	startTime    time.Time
	endTime		 time.Time
	requestsSize int
	responseSize int
	client       net.IP
	podIP        string
}

func ReverseProxyHandler(ctx *fasthttp.RequestCtx) {
	req := &ctx.Request

	resp := &ctx.Response
	client := ctx.RemoteIP()
	requestSize := len(req.Body()) + len(req.URI().QueryString())

	tStart := time.Now()
	prepareRequest(req, ctx)

	if err := proxyClient.Do(req, resp); err != nil {
		resp.SetStatusCode(fasthttp.StatusBadGateway)
		ctx.Logger().Printf("error when proxying the request: %s", err)
	}

	responseSize := resp.Header.ContentLength()
	postprocessResponse(resp, ctx)
	tEnd := time.Now()

	// fix inconsistent URL with channels
	promChan <- PromMetric{
		path:         req.RequestURI(),
		statusCode:   resp.StatusCode(),
		startTime:    tStart,
		endTime:      tEnd,
		requestsSize: requestSize,
		responseSize: responseSize,
		client:       client,
		podIP:        podIP,
	}

	go func(promChan chan PromMetric) {
		metric := <-promChan

		// TODO: potential bug
		if metric.responseSize < 0 {
			metric.responseSize = 0
		}

		strPath := string(metric.path)
		strPath = strings.Split(strPath, "&")[0]

		code := strconv.Itoa(metric.statusCode)
		if httpRequestsTotal != nil {
			httpRequestsTotal.With(prometheus.Labels{
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
				"node":      nodeName,
				"pod":       podName,
				"namespace": podNamespace,
				"path":      strPath,
				"code":      code,
				"src":       metric.client.String(),
				"dst":       metric.podIP,
			}).Observe(metric.endTime.Sub(metric.startTime).Seconds())
		}

		if httpSize != nil {
			// src -> dst
			httpSize.With(prometheus.Labels{
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
				"direction": "backward",
				"node":      nodeName,
				"pod":       podName,
				"namespace": podNamespace,
				"code":      code,
				"path":		 strPath,
				//"size":      strconv.Itoa(metric.responseSize),
				"src": metric.podIP,
				"dst": metric.client.String(),
			}).Add(float64(metric.responseSize))
		}

	}(promChan)
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
		req.Header.Set("X-Forwarded-For", xff + ", " + clientIp)
	}

	req.Header.Set("X-Forwarded-Host", proxyClient.Addr)

	if inTotal != nil {
		inTotal.With(prometheus.Labels{
			"node":      nodeName,
			"pod":       podName,
			"namespace": podNamespace,
			"code":      "0",
			"path":      string(ctxReq.RequestURI()), //+ p.sanitizeURLQuery(req.URL.RawQuery)
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
		resp.Header.Set("Pod", lastPod + ", " + podName)
	}


	ctxReq := &ctx.Request
	if outTotal != nil {
		outTotal.With(prometheus.Labels{
			"node":      nodeName,
			"pod":       podName,
			"namespace": podNamespace,
			"code":      strconv.Itoa(resp.StatusCode()),
			"path":       string(ctxReq.RequestURI()), //+ p.sanitizeURLQuery(req.URL.RawQuery)
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

		http.ListenAndServe(fmt.Sprintf(":%d", metricsPort), r)
	}()

	if err := fasthttp.ListenAndServe(fmt.Sprintf(":%d", proxyPort), ReverseProxyHandler); err != nil {
		log.Fatalf("error in fasthttp server: %s", err)
	}
}
