# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Serve a suite to a browser and take its report back.

The server knows nothing about browsers. It hands out the suite directory
over loopback, so the page is a normal http origin with ``fetch`` and
workers available, and it accepts the one ``POST /report`` the page makes
when it is done. Everything it saw is kept for the stderr file, because
when a run fails the request log is what says how far the page got.
"""

from __future__ import annotations

import threading
import urllib.parse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPORT_PATH = "/report"


class _Handler(SimpleHTTPRequestHandler):
    # Not on the stdlib's list on every platform, and a browser refuses to
    # instantiate wasm served as application/octet-stream.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
        ".js": "text/javascript",
    }

    server: BenchServer

    def do_POST(self):
        if urllib.parse.urlsplit(self.path).path != REPORT_PATH:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.server.received(body)

    def end_headers(self):
        # A fresh profile has no cache, but a browser that is not fresh
        # (Safari) must not serve a previous run's page either.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        self.server.requests.append(f"{self.address_string()} {format % args}")


class BenchServer(ThreadingHTTPServer):
    """``with BenchServer(suite_dir) as server:`` serves on an ephemeral
    loopback port for the block; ``url()`` names a page, ``wait_for_report``
    blocks for the page's POST."""

    daemon_threads = True

    def __init__(self, root: Path):
        super().__init__(("127.0.0.1", 0), partial(_Handler, directory=str(root)))
        self.requests: list[str] = []
        self._report: bytes | None = None
        self._reported = threading.Event()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    def __enter__(self) -> BenchServer:
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()
        self._thread.join()

    @property
    def port(self) -> int:
        return self.server_address[1]

    def url(self, page: str, **params: str) -> str:
        query = urllib.parse.urlencode(params)
        return f"http://127.0.0.1:{self.port}/{page}" + (f"?{query}" if query else "")

    def received(self, body: bytes):
        self._report = body
        self._reported.set()

    @property
    def report(self) -> bytes | None:
        return self._report

    def wait_for_report(self, timeout: float) -> bytes | None:
        """The body once it has arrived, None after ``timeout`` seconds."""
        if self._reported.wait(timeout):
            return self._report
        return None
