import http.server
import socketserver
from tools.config_tools import PATHS

PORT = 8000

with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print("Server started at http://localhost:" + str(PORT))
    httpd.serve_forever()