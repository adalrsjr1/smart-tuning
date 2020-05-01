from http.server import HTTPServer, BaseHTTPRequestHandler
import app_config
from urllib.parse import urlparse, parse_qsl

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
        self.wfile.write(result)

    def do_HEAD(self):
        path, query = self._parse_url(self.path)
        result, response_code = self.routers(path, query)
        self._set_headers(response_code)

    def do_POST(self):
        # Doesn't do anything with posted data
        path, query = self._parse_url(self.path)
        result, response_code = self.routers(path, query)
        self._set_headers(response_code)
        self.wfile.write(result)

    def routers(self, path, query):
        table = {
            '/register': do_register,
            '/unregister': do_unregister,
            '/': do_health,
            '/health': do_health,
            '/ping': do_health,
        }

        if path in table:
            return table[path](path, query)
        else:
            return ''.encode('utf8'), 404

print(f'mongo: {app_config.MONGO_ADDR}:{app_config.MONGO_PORT}')
db = app_config.client[app_config.MONGO_DB]
collection = db.register_collection
def do_register(path:str, record:dict)->str:
    try:
        return f'"{collection.insert_one(record).inserted_id}"'.encode('utf8'), 200
    except Exception as e:
        return f'{{"registered": {False}, "error":"{str(e)}"}}'.encode('utf8'), 500

def do_unregister(path:str, record:dict)->str:
    try:
        return f'{{"unregistered": {True}, "response":{collection.delete_many({"name": record["name"][0]}).raw_result["n"]}}}'.encode('utf8'), 200
    except Exception as e:
        return f'{{"unregistered": {False}, "error":"{str(e)}"}}'.encode('utf8'), 500

def do_health(path, record):
    return f'{{"up": {True}}}'.encode('utf8'), 200

def list():
    pipeline = [
        {"$unwind": "$name"},
        {"$group": {"_id": {
            "name":"$name",
            "ip":"$ip",
        }, "count": {"$sum": 1}}}
    ]
    return {__parse_podname(item['_id']['name']): item['_id']['ip'] for item in collection.aggregate(pipeline)}

def __parse_podname(name):
    return '-'.join(name.split('-')[:-2])

httpd = None
def run(addr="0.0.0.0", port=5000):
    server_address = (addr, port)
    global httpd
    httpd = HTTPServer(server_address, SimpleServer)

    print(f"Starting register server on {addr}:{port}")
    httpd.serve_forever()

def start():
    app_config.executor.submit(run, '0.0.0.0', 5000)

if __name__ == '__main__':
    endpoints = list()
    for endpoint in endpoints.values():
        print(endpoint)
    # start()