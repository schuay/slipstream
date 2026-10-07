# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Serve a suite to a browser and take its report back.

The server knows nothing about browsers. It hands out the suite directory
over loopback, so the page is a normal http origin with ``fetch`` and
workers available, and it accepts the one ``POST /report`` the page makes
when it is done. Everything it saw is kept for the stderr file, because
when a run fails the request log is what says how far the page got.

A suite whose page cannot report on its own (Speedometer) names an
``inject`` script. The server then serves that script, bundled with
slipstream, under ``/__slipstream/`` and appends a module tag for it to
the one ``page`` the browser is sent to -- only that page, since the
suites' own frames are the workload being measured. The checkout itself
is never written to.
"""

from __future__ import annotations

import threading
import urllib.parse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files as pkg_files
from pathlib import Path

REPORT_PATH = "/report"
INJECT_PREFIX = "/__slipstream/"
_DATA = pkg_files("slipstream.data")


def inject_tag(script: str) -> bytes:
    return f'<script type="module" src="{INJECT_PREFIX}{script}"></script>'.encode()


def injected(html: bytes, script: str) -> bytes:
    """``html`` with the module tag before ``</body>``, or appended when
    the page has no such tag; a module runs after the page's own modules
    either way, which is what the script relies on."""
    tag = inject_tag(script)
    marker = b"</body>"
    at = html.lower().rfind(marker)
    if at < 0:
        return html + b"\n" + tag + b"\n"
    return html[:at] + tag + b"\n" + html[at:]


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

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        inject = self.server.inject
        if inject is not None and path == INJECT_PREFIX + inject:
            self._send_bytes(_DATA.joinpath(inject).read_bytes(), "text/javascript")
            return
        if inject is not None and path == "/" + self.server.page:
            page = Path(self.directory) / self.server.page
            try:
                html = page.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(injected(html, inject), "text/html")
            return
        super().do_GET()

    def _send_bytes(self, body: bytes, content_type: str):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
    blocks for the page's POST. ``page`` and ``inject`` are the suite's:
    the script to append, and the one page it is appended to."""

    daemon_threads = True
    # TCPServer's default backlog is only five. Bursts of workload scripts
    # can overflow it on macOS, resetting connections before their GETs ever
    # reach the handler and leaving benchmark applications partly loaded.
    request_queue_size = 128

    def __init__(
        self, root: Path, *, page: str = "index.html", inject: str | None = None
    ):
        super().__init__(("127.0.0.1", 0), partial(_Handler, directory=str(root)))
        self.page = page
        self.inject = inject
        self.requests: list[str] = []
        self._report: bytes | None = None
        self._reported = threading.Event()
        # shutdown() returns only when serve_forever's select loop wakes, so
        # its poll interval is the teardown latency: the default 0.5s is
        # paid at the end of every run and, more visibly, every test.
        self._thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )

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

    def page_url(self, page: str, query: str) -> str:
        """``page`` with a query string spelled as the suite spells it."""
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
