Build instructions:

```bash
nvm install latest
npm install
npx webpack -w
```

Generate key pair for SSL
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
```

Start https server
```python
import http.server
import ssl


def get_ssl_context(certfile, keyfile):
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain(certfile, keyfile)
    context.set_ciphers("@SECLEVEL=1:ALL")
    return context


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        print(post_data.decode("utf-8"))


server_address = ("127.0.0.1", 5000)
httpd = http.server.HTTPServer(server_address, MyHandler)

context = get_ssl_context("cert.pem", "key.pem")
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

httpd.serve_forever()
```

Note that to test WebGPU stuff on linux the best option seems to be to download firefox nightly.
