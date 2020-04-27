from http.server import HTTPServer, BaseHTTPRequestHandler
import app_config
from urllib.parse import urlparse, parse_qs

from pymongo import MongoClient

class SimpleServer(BaseHTTPRequestHandler):
    def _parse_url(self, path):
        return urlparse(self.path).path, parse_qs(urlparse(path).query)

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
            return table[path](path, query), 200
        else:
            return ''.encode('utf8'), 404

client = MongoClient(app_config.MONGO_ADDR, app_config.MONGO_PORT)
print(f'mongo: {app_config.MONGO_ADDR}:{app_config.MONGO_PORT}')
db = client[app_config.REGISTER_DB]
collection = db.register_collection
from bson.objectid import ObjectId
def do_register(path:str, record:dict)->str:
    try:
        return f'{{"registered": {True}, "id":{collection.insert_one(record).inserted_id}}}'.encode('utf8')
    except Exception as e:
        return f'{{"registered": {False}, "error":"{str(e)}"}}'.encode('utf8')

def do_unregister(path:str, record:dict)->str:
    try:
        return f'{{"unregistered": {True}, "response":{collection.delete_one({"_id": ObjectId(record["_id"][0])}).raw_result}}}'.encode('utf8')
    except Exception as e:
        return f'{{"unregistered": {False}, "error":"{str(e)}"}}'.encode('utf8')

def do_health():
    return f'{{"up": {True}}}'.encode('utf8')

def list():
    return [item for item in collection.find()]


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
    print(list())
    # start()