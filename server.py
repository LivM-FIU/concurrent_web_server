"""
Concurrent Web Server
Author: Livan Miranda
Description:
  Multithreaded web server handling multiple simultaneous clients.
  Serves static files and LLM-based recommendation API.

Phase 2 - Step 1: Enhanced Concurrency Version
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

# ─────────────────────────────
#  Logging Configuration
# ─────────────────────────────
LOG_FILE = os.path.join("logs", "app.log")
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),  # ← prints to console
    ],
)
# ─────────────────────────────
#  Global Variables
# ─────────────────────────────
active_connections = 0
connection_lock = threading.Lock()
rate_limit_lock = threading.Lock()
shutdown_event = threading.Event()
request_history = defaultdict(deque)

RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60  # seconds

# ─────────────────────────────
#  Utility Functions
# ─────────────────────────────
def send_response(conn, code, body, content_type="application/json", headers=None):
    """Formats and sends an HTTP response."""
    status = HTTPStatus(code)
    encoded_body = body.encode("utf-8")
    header_lines = [
        f"HTTP/1.1 {code} {status.phrase}",
        f"Content-Type: {content_type}; charset=utf-8",
        f"Content-Length: {len(encoded_body)}",
        "Connection: close",
    ]

    if headers:
        for key, value in headers.items():
            header_lines.append(f"{key}: {value}")

    response_head = "\r\n".join(header_lines).encode("utf-8") + b"\r\n\r\n"
    conn.sendall(response_head + encoded_body)


def parse_request(request_data):
    """Splits an HTTP request into method, path, and body."""
    try:
        lines = request_data.split("\r\n")
        method, path, _ = lines[0].split(" ")
        body = request_data.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in request_data else ""
        return method, path, body
    except Exception:
        return None, None, None


def serve_static_file(path):
    """Serves files from /static directory."""
    file_path = os.path.join("static", path.lstrip("/"))
    if not os.path.exists(file_path):
        return 404, "<h1>404 Not Found</h1>", "text/html"

    with open(file_path, "r", encoding="utf-8") as f:
        body = f.read()

    # MIME type inference (simple)
    if file_path.endswith(".html"):
        mime = "text/html"
    elif file_path.endswith(".css"):
        mime = "text/css"
    elif file_path.endswith(".js"):
        mime = "application/javascript"
    else:
        mime = "text/plain"
    return 200, body, mime


def is_rate_limited(ip_address):
    """Simple fixed-window rate limiter per IP."""
    now = time()
    with rate_limit_lock:
        history = request_history[ip_address]
        while history and now - history[0] > RATE_LIMIT_WINDOW:
            history.popleft()

        if len(history) >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW - (now - history[0])))
            return True, retry_after

        history.append(now)
        return False, None


# ─────────────────────────────
#  Client Handler
# ─────────────────────────────
def handle_client(conn, addr):
    global active_connections
    with connection_lock:
        active_connections += 1
        logging.info(f"[+] Connection from {addr} | Active: {active_connections}")

    try:
        request = conn.recv(8192).decode("utf-8")
        if not request:
            return

        method, path, body = parse_request(request)
        if not method:
            send_response(conn, 400, json.dumps({"error": "Malformed request"}))
            return

        limited, retry_after = is_rate_limited(addr[0])
        if limited:
            headers = {"Retry-After": str(retry_after)} if retry_after else None
            send_response(
                conn,
                429,
                json.dumps({"error": "Too Many Requests", "retry_after": retry_after}),
                headers=headers,
            )
            return

        # Root or static file serving
        if method == "GET" and (path == "/" or path.startswith("/static")):
            if path == "/":
                path = "index.html"
            else:
                path = path.replace("/static/", "")
            code, content, mime = serve_static_file(path)
            send_response(conn, code, content, mime)
            return

        # API endpoint for recommendations
        elif method == "POST" and path == "/api/recommend":
            try:
                data = json.loads(body or "{}")
            except json.JSONDecodeError:
                send_response(conn, 400, json.dumps({"error": "Invalid JSON payload"}))
                return
            prompt = data.get("prompt", "")
            result = llm_recommender(prompt)
            send_response(conn, 200, json.dumps(result))
            return

        # Health/status endpoint
        elif method == "GET" and path == "/api/status":
            status_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_connections": active_connections,
                "status": "running",
            }
            send_response(conn, 200, json.dumps(status_data))
            return

        # Unrecognized endpoint
        send_response(conn, 404, json.dumps({"error": "Not Found"}))

    except Exception as e:
        logging.exception(f"Error handling {addr}: {e}")
        send_response(conn, 500, json.dumps({"error": "Internal Server Error"}))

    finally:
        conn.close()
        with connection_lock:
            active_connections -= 1
            logging.info(f"[-] Connection closed {addr} | Active: {active_connections}")


# ─────────────────────────────
#  Graceful Shutdown
# ─────────────────────────────
def handle_shutdown(sig, frame):
    logging.info("Shutdown signal received. Stopping server...")
    shutdown_event.set()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# ─────────────────────────────
#  Server Main Loop
# ─────────────────────────────
def start_server():
    host, port = settings.HOST, settings.PORT
    logging.info(f"Server starting on {host}:{port} ...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(50)  # up to 50 simultaneous waiting connections
        logging.info("Server ready and listening for connections.")

        while not shutdown_event.is_set():
            try:
                conn, addr = server.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                thread.start()
            except OSError:
                break  # triggered by shutdown_event

if __name__ == "__main__":
    start_server()
