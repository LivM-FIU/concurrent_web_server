import socket
import threading
import json
import logging
from config.settings import settings
from llm_recommender.recommender import llm_recommender

# Logging setup
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def send_response(conn, code, body, content_type="application/json"):
    response = (
        f"HTTP/1.1 {code} OK\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
        f"{body}"
    )
    conn.sendall(response.encode())

def handle_client(conn, addr):
    try:
        request = conn.recv(4096).decode()
        if not request:
            return
        method, path, _ = request.split(' ', 2)
        logging.info(f"Request from {addr}: {method} {path}")

        if method == 'GET' and path == '/':
            body = "<h2>Concurrent Web Server running.</h2>"
            send_response(conn, 200, body, "text/html")

        elif method == 'POST' and path == '/api/recommend':
            body = request.split('\r\n\r\n')[1]
            data = json.loads(body)
            result = llm_recommender(data.get("prompt", ""))
            send_response(conn, 200, json.dumps(result))

        else:
            send_response(conn, 404, json.dumps({"error": "Not Found"}))
    except Exception as e:
        logging.error(f"Error handling client {addr}: {e}")
        send_response(conn, 500, json.dumps({"error": "Internal Server Error"}))
    finally:
        conn.close()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((settings.HOST, settings.PORT))
        server.listen()
        logging.info(f"Server listening on {settings.HOST}:{settings.PORT}")
        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr)).start()

if __name__ == "__main__":
    start_server()
