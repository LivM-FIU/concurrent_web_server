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
from datetime import datetime
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
lock = threading.Lock()
shutdown_event = threading.Event()

# ─────────────────────────────
#  Utility Functions
# ─────────────────────────────
def send_response(conn, code, body, content_type="application/json"):
    """Formats and sends an HTTP response."""
    response = (
        f"HTTP/1.1 {code} OK\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
        f"{body}"
    )
    conn.sendall(response.encode("utf-8"))


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


# ─────────────────────────────
#  Client Handler
# ─────────────────────────────
def handle_client(conn, addr):
    global active_connections
    with lock:
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
            data = json.loads(body or "{}")
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
        with lock:
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
