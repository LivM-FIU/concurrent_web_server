"""
server.py — Concurrent Web Server
"""

import socket
import threading
import json
import logging
import os
import signal
import sys
from collections import defaultdict, deque
from datetime import datetime
from http import HTTPStatus
from time import time

from config.settings import settings
from llm_recommender.recommender import llm_recommender

# -------------------------
# Logging
# -------------------------
LOG_FILE = os.path.join("logs", "app.log")
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

# -------------------------
# Server State
# -------------------------
active_connections = 0
connection_lock = threading.Lock()

rate_limit_lock = threading.Lock()
request_history = defaultdict(deque)

shutdown_event = threading.Event()

RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60


# -------------------------
# HTTP Utility
# -------------------------
def send_response(conn, code, body, content_type="application/json", headers=None):
    status = HTTPStatus(code)

    if isinstance(body, str):
        encoded_body = body.encode("utf-8")
    else:
        encoded_body = body

    header_lines = [
        f"HTTP/1.1 {code} {status.phrase}",
        f"Content-Type: {content_type}; charset=utf-8",
        f"Content-Length: {len(encoded_body)}",
        "Connection: close",
        "Access-Control-Allow-Origin: *",
        "Access-Control-Allow-Headers: Content-Type",
        "Access-Control-Allow-Methods: GET, POST, OPTIONS",
    ]

    if headers:
        for k, v in headers.items():
            header_lines.append(f"{k}: {v}")

    response_bytes = ("\r\n".join(header_lines) + "\r\n\r\n").encode("utf-8") + encoded_body

    try:
        conn.sendall(response_bytes)
    except:
        pass


def parse_request(request_data):
    try:
        header_part, _, body = request_data.partition("\r\n\r\n")
        header_lines = header_part.split("\r\n")

        method, path, _ = header_lines[0].split(" ", 2)

        headers = {}
        for line in header_lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.lower()] = v.strip()

        return method, path, headers, body
    except:
        return None, None, {}, ""


# -------------------------
# Static File Serving (FIXED)
# -------------------------
def serve_static(path):
    """
    FIXED:
    - Correctly maps /static/... → static/...
    - Prevents static/static/... bug
    """

    if path.startswith("/static/"):
        relative = path[len("/static/"):]  # strip leading folder
        file_path = os.path.join("static", relative)
    else:
        # Fallback (not normally used)
        file_path = os.path.join("static", path.lstrip("/"))

    if not os.path.exists(file_path):
        return 404, b"<h1>404 Not Found</h1>", "text/html"

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except:
        return 500, b"<h1>500 Internal Server Error</h1>", "text/html"

    # MIME types
    ext = file_path.lower()
    if ext.endswith(".html"):
        mime = "text/html"
    elif ext.endswith(".css"):
        mime = "text/css"
    elif ext.endswith(".js"):
        mime = "application/javascript"
    elif ext.endswith(".png"):
        mime = "image/png"
    elif ext.endswith(".jpg") or ext.endswith(".jpeg"):
        mime = "image/jpeg"
    else:
        mime = "application/octet-stream"

    return 200, data, mime


# -------------------------
# Rate Limiting
# -------------------------
def is_rate_limited(ip):
    now = time()
    with rate_limit_lock:
        q = request_history[ip]

        while q and now - q[0] > RATE_LIMIT_WINDOW:
            q.popleft()

        if len(q) >= RATE_LIMIT_REQUESTS:
            retry_after = int(RATE_LIMIT_WINDOW - (now - q[0]))
            return True, retry_after

        q.append(now)
        return False, None


# -------------------------
# Client Handler
# -------------------------
def handle_client(conn, addr):
    global active_connections

    with connection_lock:
        active_connections += 1
        logging.info(f"[+] {addr} connected | Active: {active_connections}")

    try:
        raw = conn.recv(50000).decode("utf-8", errors="ignore")
        if not raw:
            return

        method, path, headers, body = parse_request(raw)

        if method == "OPTIONS":
            send_response(conn, 200, "")
            return

        # Rate limit
        limited, retry_after = is_rate_limited(addr[0])
        if limited:
            send_response(
                conn,
                429,
                json.dumps({"error": "Too Many Requests", "retry_after": retry_after}),
                headers={"Retry-After": str(retry_after)},
            )
            return

        # Serve static files (fully working)
        if method == "GET":
            if path == "/":
                path = "/static/index.html"

            if path.startswith("/static/"):
                code, content, mime = serve_static(path)
                send_response(conn, code, content, mime)
                return

        # Recommendation API
        if method == "POST" and path == "/api/recommend":
            try:
                data = json.loads(body or "{}")
            except:
                send_response(conn, 400, json.dumps({"error": "Invalid JSON"}))
                return

            prompt = data.get("prompt", "")
            user_id = data.get("user_id", "anonymous")

            result = llm_recommender(prompt, user_id=user_id)
            send_response(conn, 200, json.dumps(result))
            return

        # Health check
        if method == "GET" and path == "/api/status":
            send_response(conn, 200, json.dumps({
                "status": "running",
                "timestamp": datetime.utcnow().isoformat(),
                "active_connections": active_connections,
            }))
            return

        send_response(conn, 404, json.dumps({"error": "Not Found"}))

    except Exception as e:
        logging.exception("Handler Error: %s", e)
        send_response(conn, 500, json.dumps({"error": "Internal Server Error"}))

    finally:
        try: conn.close()
        except: pass

        with connection_lock:
            active_connections -= 1
            logging.info(f"[-] {addr} disconnected | Active: {active_connections}")


# -------------------------
# Shutdown
# -------------------------
def handle_shutdown(sig, frame):
    logging.info(f"Shutdown signal received: {sig}")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# -------------------------
# Server Start
# -------------------------
def start_server():
    host, port = settings.HOST, settings.PORT
    logging.info(f" Server starting on {host}:{port}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(128)
        logging.info(" Server is ready.")

        while not shutdown_event.is_set():
            try:
                conn, addr = server.accept()
                threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
            except Exception as exc:
                logging.exception("Accept Loop Error: %s", exc)
                break

    logging.info(" Server terminated.")


if __name__ == "__main__":
    start_server()
