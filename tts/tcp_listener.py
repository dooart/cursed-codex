"""TCP client with auto-reconnect for the Codex event tap."""

import json
import os
import socket
import threading
import time

SOCK_FILE = os.path.join(os.path.expanduser("~"), ".codex", "commentator.sock")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4222


def _resolve_addr():
    """Read address from sock file, fall back to default."""
    try:
        with open(SOCK_FILE) as f:
            raw = f.read().strip()
        if raw:
            raw = raw.removeprefix("tcp://").removeprefix("ws://")
            host, _, port_str = raw.partition(":")
            return (host or DEFAULT_HOST, int(port_str) if port_str else DEFAULT_PORT)
    except (OSError, ValueError):
        pass
    return (DEFAULT_HOST, DEFAULT_PORT)


class TcpListener:
    """Connects to Codex TCP tap, parses newline-delimited JSON, calls back with events."""

    def __init__(self, on_event):
        self._on_event = on_event
        self._stopped = False
        self._reconnect_delay = 1.0

    def start(self):
        self._stopped = False
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stopped = True

    def _run(self):
        while not self._stopped:
            host, port = _resolve_addr()
            try:
                self._connect(host, port)
            except Exception as e:
                if self._stopped:
                    return
                print(f"TCP connection error: {e}")
            # Reconnect with exponential backoff
            if not self._stopped:
                print(f"Reconnecting in {self._reconnect_delay:.0f}s...")
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 10.0)

    def _connect(self, host, port):
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.settimeout(None)  # recv blocks indefinitely once connected
            print(f"Connected to Codex at {host}:{port}")
            self._reconnect_delay = 1.0
            buf = b""
            while not self._stopped:
                data = sock.recv(4096)
                if not data:
                    print("TCP connection closed by server")
                    return
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._handle_line(line)

    def _handle_line(self, line):
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        msg = data.get("msg")
        if isinstance(msg, dict) and msg.get("type"):
            self._on_event(msg)
