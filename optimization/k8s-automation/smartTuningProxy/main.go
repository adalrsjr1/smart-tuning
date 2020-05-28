package main

// Source:
// https://venilnoronha.io/hand-crafting-a-sidecar-proxy-and-demystifying-istio

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"io"
	"net/http"
	"net/http/pprof"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"

	//"github.com/prometheus/client_golang/prometheus"
	//"github.com/prometheus/client_golang/prometheus/promauto"
	//"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	proxyPort, _   = strconv.Atoi(os.Getenv("PROXY_PORT"))
	servicePort, _ = strconv.Atoi(os.Getenv("SERVICE_PORT"))
	metricsPort, _ = strconv.Atoi(os.Getenv("METRICS_PORT"))
	proxyURL 	   = "http://127.0.0.1:" + os.Getenv("SERVICE_PORT")

	// https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/
	podName           = os.Getenv("POD_NAME")
	nodeName          = os.Getenv("NODE_NAME")
	podNamespace      = os.Getenv("POD_NAMESPACE")
	svcName			  = os.Getenv("HOST_IP")
	podIP             = os.Getenv("POD_IP")
	podServiceAccount = os.Getenv("POD_SERVICE_ACCOUNT")

	httpClient 		  = http.Client{}

	//httpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	//	// when change this name also update prometheus config file
	//	Name: "remap_http_requests_total",
	//	Help: "Count of all HTTP requests",
	//}, []string{"node", "pod", "namespace", "code", "method", "path"})

	keysBuffer = make([]string, 1000)
)

// Create a structure to define the proxy functionality.
type Proxy struct{}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Forward the HTTP request to the destination service
	res, err := p.forwardRequest(req)

	// Notify the client if there was an error while forwarding the request.
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	// If the request was forwared successfully, write the response back to the client.
	p.writeResponse(w, res)

	//go func(req *http.Request, res *http.Response) {
	//	// chopping of all queries
	//	httpRequestsTotal.With(prometheus.Labels{
	//		"node":      nodeName,
	//		"pod":       podName,
	//		"namespace": podNamespace,
	//		"code":      strconv.Itoa(res.StatusCode),
	//		"method":    req.Method,
	//		"path":      req.URL.Path, //+ p.sanitizeURLQuery(req.URL.RawQuery),
	//	}).Inc()
	//}(req, res)
}

func (p *Proxy) sanitizeURLQuery(rawQuery string) string {
	query, _ := url.ParseQuery(rawQuery)
	keys := keysBuffer[0:len(query)]
	i := 0
	for k := range query {
		keys[i] = k
		i++
	}

	sort.Strings(keys)

	var strQuery strings.Builder

	for _, key := range keys {
		strQuery.WriteString(key)
		strQuery.WriteString(",")
	}

	if i > 0 {
		return "?" + strQuery.String()
	}
	return ""
}

func (p *Proxy) forwardRequest(req *http.Request) (*http.Response, error) {
	proxyReq, err := http.NewRequest(req.Method, proxyURL+req.RequestURI, req.Body)

	//proxyReq.Header = req.Header
	p.copyHeader(req.Header, proxyReq.Header)
	proxyReq.Header.Set("Server", "smarttuning-proxy")
	proxyReq.Header.Set("POD", podName)

	// Capture the duration while making a request to the destination service.
	res, err := httpClient.Do(proxyReq)

	// Return the response, the request duration, and the error.
	return res, err
}

func (p *Proxy) writeResponse(w http.ResponseWriter, res *http.Response) {
	// Copy all the header values from the response.
	p.copyHeader(res.Header, w.Header())

	// Set a special header to notify that the proxy actually serviced the
	// request.
	w.Header().Set("Server", "X-SMART-TUNING-PROXY")
  	w.Header().Set("Pod", podName)

	// Set the status code returned by the destination service.
	w.WriteHeader(res.StatusCode)

	// Copy the contents from the response body.
	io.Copy(w, res.Body)

	// Finish the request.
	res.Body.Close()
}

func (p *Proxy) copyHeader(sourceHeader, destinationHeader http.Header) {
	for name, values := range sourceHeader {
		destinationHeader[name] = values
	}
}


/*
no proxy : 1426.8
custom   : 919.5
envoy    : 1312.2
traefik  : 1011.8

cat jmeter/$(ls -lha jmeter | tail -n 1 | awk '{print $9}')/*.csv | tail -n 1 | awk -F ',' '{print $11}'
 */
func main() {
  	fmt.Printf("Initalizing proxy")

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
	// Listen on the predefined proxy port.
	fmt.Sprintf("0.0.0.0:%d/ --> 127.0.0.1:%d", proxyPort, servicePort)
	http.ListenAndServe(fmt.Sprintf(":%d", proxyPort), &Proxy{})

}

