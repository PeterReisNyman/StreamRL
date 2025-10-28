#!/usr/bin/env python3
"""

  - http://127.0.0.1:8000/index.html?room=alpha&name=Alice
  - http://127.0.0.1:8000/index.html?room=alpha&name=Bob

  
Launches the Voxelcraft demo by starting a simple HTTP server in the
voxelcraft directory and opening the browser to the correct URL.

Usage:
  python3 run_voxelcraft.py

Notes:
  - If port 8000 is busy, it will try the next available port.
  - Press Ctrl+C to stop the server.
"""
import argparse
import contextlib
import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser


def find_open_port(start=8000, end=8100, host="127.0.0.1"):
    """Find an available TCP port between start and end (inclusive)."""
    for port in range(start, end + 1):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No open port found in range {start}-{end}")


class ReuseTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


SSE_CLIENTS = {}  # room -> set(handlers)
WORLD_ROOMS = {}  # room -> { seed, edits, clients }


def room_state(room: str):
    return WORLD_ROOMS.setdefault(room, {"seed": None, "edits": {}, "clients": {}})


def sse_broadcast(room: str, payload: dict):
    data = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
    clients = SSE_CLIENTS.get(room, set())
    dead = []
    for h in list(clients):
        try:
            h.wfile.write(data)
            h.wfile.flush()
        except Exception:
            dead.append(h)
    for h in dead:
        try:
            clients.discard(h)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Serve the Voxelcraft demo")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: auto from 8000..8100)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    vc_dir = os.path.join(repo_root, "voxelcraft")
    index_path = os.path.join(vc_dir, "index.html")

    if not os.path.isdir(vc_dir) or not os.path.isfile(index_path):
        print("Error: voxelcraft/index.html not found. Run this from the repo that contains the 'voxelcraft' folder.")
        sys.exit(1)

    os.chdir(vc_dir)
    host = args.host
    if args.port is not None:
        ports_to_try = [args.port]
    else:
        # Try the default range and be resilient to races by retrying on bind failure
        ports_to_try = list(range(8000, 8101))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # quiet output
            sys.stdout.write("[srv] " + fmt % args + "\n")

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            room = (qs.get('room', ['default'])[0]) or 'default'
            if self.path.startswith("/events"):
                # SSE subscription
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                SSE_CLIENTS.setdefault(room, set()).add(self)
                self._room = room
                # heartbeat to keep connection open
                try:
                    while True:
                        time.sleep(15)
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except Exception:
                            break
                finally:
                    try:
                        SSE_CLIENTS.get(room, set()).discard(self)
                    except Exception:
                        pass
                return
            if self.path.startswith("/snapshot"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(room_state(room)).encode("utf-8"))
                return
            return super().do_GET()

        def do_POST(self):
            if self.path.startswith("/publish"):
                length = int(self.headers.get('content-length', '0') or 0)
                body = self.rfile.read(length) if length else b"{}"
                try:
                    evt = json.loads(body.decode("utf-8"))
                except Exception:
                    evt = {}
                # update world state minimally
                t = evt.get("type")
                cid = evt.get("clientId")
                room = evt.get("room") or 'default'
                state = room_state(room)
                if t == "seed":
                    if not state["seed"]:
                        state["seed"] = evt.get("seed")
                elif t == "pos" and cid:
                    state["clients"][cid] = {
                        "x": evt.get("x"),
                        "y": evt.get("y"),
                        "z": evt.get("z"),
                        "color": evt.get("color"),
                        "name": evt.get("name"),
                        "yaw": evt.get("yaw"),
                        "pitch": evt.get("pitch"),
                    }
                elif t == "edit":
                    key = evt.get("key")
                    bid = evt.get("id")
                    if isinstance(key, str) and isinstance(bid, int):
                        if bid == 0:
                            state["edits"].pop(key, None)
                        else:
                            state["edits"][key] = bid
                elif t == "leave" and cid:
                    state["clients"].pop(cid, None)
                # broadcast to all subscribers
                sse_broadcast(room, evt)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            if self.path.startswith("/save_screenshot"):
                length = int(self.headers.get('content-length', '0') or 0)
                body = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(body.decode("utf-8"))
                    data_url = payload.get("dataUrl", "")
                except Exception:
                    data_url = ""
                # Expect data URL like data:image/png;base64,....
                import base64, time
                ok = False
                filename = None
                if isinstance(data_url, str) and data_url.startswith("data:image/png;base64,"):
                    try:
                        b64 = data_url.split(",", 1)[1]
                        raw = base64.b64decode(b64)
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"Screenshot-{ts}.png"
                        with open(filename, "wb") as out:
                            out.write(raw)
                        ok = True
                    except Exception:
                        ok = False
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = {"ok": ok, "file": filename}
                self.wfile.write(json.dumps(resp).encode("utf-8"))
                return
            self.send_response(404)
            self.end_headers()

    handler = Handler
    httpd = None
    last_err = None
    for port in ports_to_try:
        try:
            httpd = ReuseTCPServer((host, port), handler)
            break
        except OSError as e:
            last_err = e
            continue
    if httpd is None:
        raise SystemExit(f"Failed to bind server on any port {ports_to_try[0]}..{ports_to_try[-1]}: {last_err}")

    url = f"http://{host}:{port}/index.html"
    print(f"Serving Voxelcraft on {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
