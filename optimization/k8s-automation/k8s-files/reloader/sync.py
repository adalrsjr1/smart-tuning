import kubernetes as k8s
import reloader
import os, sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qsl
import traceback

class SimpleServer(BaseHTTPRequestHandler):
    def _parse_url(self, path):
        return urlparse(self.path).path, dict(parse_qsl(urlparse(path).query))

    def _set_headers(self, response):
        self.send_response(response)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_GET(self):
        path, query = self._parse_url(self.path)
        result, response_code = self.routers(path, query)
        self._set_headers(response_code)
        self.wfile.write(str(result).encode('utf-8'))

    def do_HEAD(self):
        path, query = self._parse_url(self.path)
        result, response_code = self.routers(path, query)
        self._set_headers(response_code)

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length) # <--- Gets the data itself
            path, query = self._parse_url(self.path)
            post_data = dict(parse_qsl(post_data))
            print(f'path:{path}, query:{query}, post_data:{post_data}')
            result, response_code = self.routers(path, post_data)
            print(f'result:{result}, response_code:{response_code}')
            self._set_headers(response_code)
            self.wfile.write(str(result).encode('utf-8'))
        except Exception as e:
            traceback.print_exc()

    def routers(self, path, query):
        table = {
            '/reload': do_reload,
            '/': do_health,
            '/health': do_health,
            '/ping': do_health,
        }

        if path in table:
            return table[path](path, query)
        else:
            return ''.encode('utf8'), 404

def do_reload(path:str, record)->str:
    try:
        print('reloading...')
        configMap = ConfigMap()
        return configMap.patch(reloader.configmap_name, reloader.namespace, record), 200
    except Exception as e:
        traceback.print_exc()
        return e, 500


def do_health(path, record):
    return f'{{"up": {True}}}'.encode('utf8')

class ConfigMap:
    def __init__(self):
        k8s.config.load_incluster_config()

    def patch(self, configmap_name, configmap_namespace, data):
        # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#patch_namespaced_config_map
        data = {key.decode(): val.decode() for key, val in data.items()}
        body = {
            "kind": "ConfigMap",
            "apiVersion": "v1",
            "data": data
        }
        print(f'config: {configmap_name}, namespace: {configmap_namespace}, body: {body}')
        response = None
        try:
            api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())
            name = configmap_name
            namespace = configmap_namespace
            pretty = 'true'

            response = api_instance.patch_namespaced_config_map(name, namespace, body, pretty=pretty)
            print(response)
        except Exception as e:
            traceback.print_exc()

        return response



httpd = None
def run(addr="0.0.0.0", port=5000):
    server_address = (addr, port)
    global httpd
    httpd = HTTPServer(server_address, SimpleServer)

    print(f"Starting sync server on {addr}:{port}")
    httpd.serve_forever()

def start_server():
    reloader.executor.submit(run, '0.0.0.0', 5000)

if __name__ == '__main__':
    try:
        start_server()
        reloader.main()
    except Exception as e:
        print(e)
