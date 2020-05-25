package main

// Source:
// https://venilnoronha.io/hand-crafting-a-sidecar-proxy-and-demystifying-istio

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	proxyPort, _   = strconv.Atoi(os.Getenv("PROXY_PORT"))
	servicePort, _ = strconv.Atoi(os.Getenv("SERVICE_PORT"))
	metricsPort, _ = strconv.Atoi(os.Getenv("METRICS_PORT"))

	// https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/
	podName           = os.Getenv("POD_NAME")
	nodeName          = os.Getenv("NODE_NAME")
	podNamespace      = os.Getenv("POD_NAMESPACE")
	svcName			  = os.Getenv("HOST_IP")
	podIP             = os.Getenv("POD_IP")
	podServiceAccount = os.Getenv("POD_SERVICE_ACCOUNT")

	httpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		// when change this name also update prometheus config file
		Name: "remap_http_requests_total",
		Help: "Count of all HTTP requests",
	}, []string{"node", "pod", "namespace", "code", "method", "path"})
)

// Create a structure to define the proxy functionality.
type Proxy struct{}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Forward the HTTP request to the destination service
	res, _, err := p.forwardRequest(req)

	// Notify the client if there was an error while forwarding the request.
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	// If the request was forwared successfully, write the response back to the
	// client.
	p.writeResponse(w, res)

	// Print request and response statistics.
	httpRequestsTotal.With(prometheus.Labels{
		"node":      nodeName,
		"pod":       podName,
		"namespace": podNamespace,
		"code":      fmt.Sprintf("%d", res.StatusCode),
		"method":    req.Method,
		"path":      req.URL.Path + p.sanitizeURLQuery(req.URL.RawQuery),
	}).Inc()
}

func (p *Proxy) sanitizeURLQuery(rawQuery string) string {
	query, _ := url.ParseQuery(rawQuery)
	keys := make([]string, len(query))
	i := 0
	for k := range query {
		keys[i] = k
		i++
	}
	sort.Strings(keys)
	strQuery := strings.Join(keys, "=,")
	if i > 0 {
		return "?" + strQuery
	}
	return ""
}

func (p *Proxy) copyHeader(sourceHeader, destinationHeader http.Header) {
	for name, values := range sourceHeader {
		destinationHeader[name] = values
	}
}

func (p *Proxy) forwardRequest(req *http.Request) (*http.Response, time.Duration, error) {
	// Prepare the destination endpoint to forward the request to.
	proxyURL := fmt.Sprintf("http://127.0.0.1:%d%s", servicePort, req.RequestURI)

	// Print the original URL and the proxied request URL.
	//fmt.Printf("Original URL: http://%s/:%d%s\n", req.Host, servicePort, req.RequestURI)
	//fmt.Printf("%s:%s URL: %s\n", podName, podNamespace, proxyURL)

	// Create an HTTP client and a proxy request based on the original request.
	httpClient := http.Client{}
	proxyReq, err := http.NewRequest(req.Method, proxyURL, req.Body)
	p.copyHeader(req.Header, proxyReq.Header)
	//proxyReq.Header = req.Header

	// Capture the duration while making a request to the destination service.
	start := time.Now()
	res, err := httpClient.Do(proxyReq)
	duration := time.Since(start)

	// Return the response, the request duration, and the error.
	return res, duration, err
}

func (p *Proxy) writeResponse(w http.ResponseWriter, res *http.Response) {
	// Copy all the header values from the response.
	p.copyHeader(res.Header, w.Header())
	//for name, values := range res.Header {
	//	w.Header()[name] = values
	//}

	// Set a special header to notify that the proxy actually serviced the
	// request.
	w.Header().Set("Server", "smarttuning-proxy")
  w.Header().Set("POD", podName)

	// Set the status code returned by the destination service.
	w.WriteHeader(res.StatusCode)

	// Copy the contents from the response body.
	io.Copy(w, res.Body)

	// Finish the request.
	res.Body.Close()
}

func main() {
  fmt.Printf("Initalizing proxy")
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		http.ListenAndServe(fmt.Sprintf(":%d", metricsPort), nil)
	}()
	// Listen on the predefined proxy port.
	http.ListenAndServe(fmt.Sprintf(":%d", proxyPort), &Proxy{})

}

