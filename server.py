import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from predict3 import generate_long_text, load_resources


HOST = "127.0.0.1"
PORT = 8000

SP, MODEL, MODEL_TYPE = load_resources()


class GenerateHandler(BaseHTTPRequestHandler):
    server_version = "BigramTextServer/0.1"

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json(400, {"error": "prompt must be a non-empty string"})
            return

        max_tokens = int(payload.get("max_tokens", 80))
        top_k = int(payload.get("top_k", 8))
        temperature = float(payload.get("temperature", 0.95))
        min_len_before_eos = int(payload.get("min_len_before_eos", 32))

        text = generate_long_text(
            prompt,
            SP,
            MODEL,
            MODEL_TYPE,
            max_tokens=max_tokens,
            top_k=top_k,
            temperature=temperature,
            min_len_before_eos=min_len_before_eos,
        )
        self._send_json(200, {"text": text})

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), GenerateHandler)
    print(f"Loaded model and listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
