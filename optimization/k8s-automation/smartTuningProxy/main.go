package main

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"log"
	"net/http"
	"net/http/pprof"
	"os"
	"strconv"
	"time"

	"github.com/valyala/fasthttp"
)

var (
	proxyPort, _    = strconv.Atoi(os.Getenv("PROXY_PORT"))
	metricsPort, _  = strconv.Atoi(os.Getenv("METRICS_PORT"))
	upstreamPort, _ = strconv.Atoi(os.Getenv("SERVICE_PORT"))
	upstreamAddr    = "127.0.0.1:" + os.Getenv("SERVICE_PORT")

	// https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/
	podName           = os.Getenv("POD_NAME")
	nodeName          = os.Getenv("NODE_NAME")
	podNamespace      = os.Getenv("POD_NAMESPACE")
	svcName			  = os.Getenv("HOST_IP")
	podIP             = os.Getenv("POD_IP")
	podServiceAccount = os.Getenv("POD_SERVICE_ACCOUNT")

	maxConn, _ = strconv.Atoi(os.Getenv("MAX_CONNECTIONS"))
	readBuffer, _ = strconv.Atoi(os.Getenv("READ_BUFFER_SIZE"))
	writeBuffer, _ = strconv.Atoi(os.Getenv("WRITE_BUFFER_SIZE"))
	readTimeout, _ = strconv.Atoi(os.Getenv("READ_TIMEOUT"))
	writeTimeout, _ = strconv.Atoi(os.Getenv("WRITE_TIMEOUT"))
	connDuration, _ = strconv.Atoi(os.Getenv("MAX_IDLE_CONNECTION_DURATION"))
	connTimeout, _ = strconv.Atoi(os.Getenv("MAX_CONNECTION_TIMEOUT"))

	proxyClient = &fasthttp.HostClient{
		Addr:                          upstreamAddr,
		NoDefaultUserAgentHeader:      true, // Don't send: User-Agent: fasthttp
		MaxConns:                      maxConn,
		ReadBufferSize:                readBuffer, // Make sure to set this big enough that your whole request can be read at once.
		WriteBufferSize:               writeBuffer, // Same but for your response.
		ReadTimeout:                   time.Duration(readTimeout) * time.Second,
		WriteTimeout:                  time.Duration(writeTimeout) * time.Second,
		MaxIdleConnDuration:           time.Duration(connDuration) * time.Second,
		MaxConnWaitTimeout: 		   time.Duration(connTimeout) * time.Second,
		DisableHeaderNamesNormalizing: true, // If you set the case on your headers correctly you can enable this.
	}

	httpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		// when change this name also update prometheus config file
		Name: "smarttuning_http_requests_total",
		Help: "Count of all HTTP requests",
	}, []string{"node", "pod", "namespace", "code", "path"})

	httpLatencyHist = promauto.NewSummaryVec(prometheus.SummaryOpts{
		Name: "smarttuning_http_latency_seconds",
		Help: "Server time latency",
		Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001, 1.00: 0.00},
	}, []string{"node", "pod", "namespace", "code"})
)

func ReverseProxyHandler(ctx *fasthttp.RequestCtx) {
	req := &ctx.Request
	resp := &ctx.Response

	tStart := time.Now()
	prepareRequest(req)
	if err := proxyClient.Do(req, resp); err != nil {
		resp.SetStatusCode(fasthttp.StatusBadGateway)
		ctx.Logger().Printf("error when proxying the request: %s", err)
	}
	postprocessResponse(resp)

	go func(path []byte, start time.Time, statusCode int) {
		httpRequestsTotal.With(prometheus.Labels{
			"node":      nodeName,
			"pod":       podName,
			"namespace": podNamespace,
			"code":      strconv.Itoa(statusCode),
			"path":      string(path), //+ p.sanitizeURLQuery(req.URL.RawQuery)
		}).Inc()

		httpLatencyHist.With(prometheus.Labels{
			"node":      nodeName,
			"pod":       podName,
			"namespace": podNamespace,
			"code":      strconv.Itoa(statusCode),
		}).Observe(time.Since(start).Seconds())
	}(req.RequestURI(), tStart, resp.StatusCode())
}

func prepareRequest(req *fasthttp.Request) {
	// do not proxy "Connection" header.
	req.SetHost(proxyClient.Addr)
	req.Header.Set("X-Forwarded-Host", podIP)
	//req.Header.Del("Connection")
	// strip other unneeded headers.

	// alter other request params before sending them to upstream host


}

func postprocessResponse(resp *fasthttp.Response) {
	// do not proxy "Connection" header
	//resp.Header.Del("Connection")
	resp.Header.Add("Server", "X-SMART-TUNING-PROXY")
	resp.Header.Add("Pod", podName)

	// strip other unneeded headers

	// alter other response data if needed

}

func main() {
	go func() {
		r := http.NewServeMux()
		r.Handle("/metrics", promhttp.Handler())
		r.HandleFunc("/debug/pprof/", pprof.Index)
		r.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		r.HandleFunc("/debug/pprof/profile", pprof.Profile)
		r.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		r.HandleFunc("/debug/pprof/trace", pprof.Trace)

		http.ListenAndServe(fmt.Sprintf(":%d", metricsPort), r)
	}()

	if err := fasthttp.ListenAndServe(fmt.Sprintf(":%d", proxyPort), ReverseProxyHandler); err != nil {
		log.Fatalf("error in fasthttp server: %s", err)
	}
}