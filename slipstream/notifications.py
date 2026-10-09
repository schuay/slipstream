# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Build discovery: local vnode events, or the same events streamed over SSH.

Readers only mark engines dirty. The main watch loop owns consumers and locks.
One-second waits below check shutdown; they never scan the bus on a timer.
"""

from __future__ import annotations

import json
import os
import select
import shlex
import subprocess
import threading
import time
from pathlib import Path

PROTOCOL_VERSION = 1
RETRY_SECS = 60.0


class DirectorySubscription:
    """Read-only watches, including ancestors of topics that don't exist yet."""

    def __init__(self, root: Path, engines: list[str], input_fd=None):
        if not hasattr(select, "kqueue"):
            raise RuntimeError("build notifications require macOS/BSD kqueue")
        root = root.expanduser().absolute()
        self.topics = {name: root / "topics" / "builds" / name for name in engines}
        self.queue = select.kqueue()
        self.watches = {}  # path -> (fd, device, inode)
        self.input_fd = input_fd
        try:
            if input_fd is not None:
                self.queue.control(
                    [
                        select.kevent(
                            input_fd,
                            filter=select.KQ_FILTER_READ,
                            flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                        )
                    ],
                    0,
                    0,
                )
            self._refresh()
        except BaseException:
            self.close()
            raise

    def _refresh(self):
        # Register parents first: creation racing a missing-child lookup leaves
        # an event on an already registered parent, so we will try again.
        paths = set()
        for topic in self.topics.values():
            paths.update(topic.parents)
            paths.add(topic)
        for path in sorted(paths, key=lambda p: len(p.parts)):
            old = self.watches.get(path)
            try:
                stat = path.stat()
            except FileNotFoundError:
                stat = None
            if old and stat and old[1:] == (stat.st_dev, stat.st_ino):
                continue
            if old:
                os.close(old[0])
                del self.watches[path]
            if stat is None:
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            except FileNotFoundError:
                continue
            try:
                stat = os.fstat(fd)
                event = select.kevent(
                    fd,
                    filter=select.KQ_FILTER_VNODE,
                    flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                    fflags=(
                        select.KQ_NOTE_WRITE
                        | select.KQ_NOTE_DELETE
                        | select.KQ_NOTE_RENAME
                        | select.KQ_NOTE_REVOKE
                    ),
                )
                self.queue.control([event], 0, 0)
            except BaseException:
                os.close(fd)
                raise
            self.watches[path] = (fd, stat.st_dev, stat.st_ino)

    def changes(self, timeout=1.0):
        events = self.queue.control(None, 64, timeout)
        if not events:
            return set()
        if any(event.filter == select.KQ_FILTER_READ for event in events):
            # The command has no input protocol. EOF means its SSH client left.
            raise EOFError("subscription client closed")
        if any(event.flags & select.KQ_EV_ERROR for event in events):
            raise OSError("filesystem notification failed")
        before = {name: self.watches.get(path) for name, path in self.topics.items()}
        changed_fds = {event.ident for event in events}
        self._refresh()
        return {
            name
            for name, path in self.topics.items()
            if before[name] != self.watches.get(path)
            or (before[name] and before[name][0] in changed_fds)
        }

    def close(self):
        for fd, _, _ in self.watches.values():
            os.close(fd)
        self.watches.clear()
        self.queue.close()


def serve(root, engines, should_stop, output, input_fd=None):
    """The SSH endpoint; no config, database or machine lock is needed."""
    subscription = DirectorySubscription(Path(root), engines, input_fd)
    try:

        def send(kind, names):
            output.write(
                json.dumps(
                    {
                        "version": PROTOCOL_VERSION,
                        "type": kind,
                        "engines": sorted(names),
                    }
                )
                + "\n"
            )
            output.flush()

        send("ready", engines)
        while not should_stop():
            changed = subscription.changes()
            if changed:
                send("changed", changed)
    finally:
        subscription.close()


class WorkSchedule:
    """Coalesced events plus independent git and consumer retry deadlines."""

    def __init__(self, bus_engines, git_engines, interval):
        self.condition = threading.Condition()
        self.generation = {name: 0 for name in bus_engines}
        self.pending = set()
        self.deadlines = {name: 0.0 for name in git_engines}
        self.interval = interval

    def changed(self, names):
        with self.condition:
            for name in names:
                self.generation[name] += 1
                self.pending.add(name)
            self.condition.notify_all()

    def snapshot(self, name):
        with self.condition:
            return self.generation[name]

    def finished(self, name, generation, *, more=False, retry=None):
        with self.condition:
            if retry is not None:
                self.pending.add(name)
                self.deadlines[name] = time.monotonic() + retry
            else:
                self.deadlines.pop(name, None)
                if more or self.generation[name] != generation:
                    self.pending.add(name)
                else:
                    self.pending.discard(name)

    def git_finished(self, name):
        with self.condition:
            self.deadlines[name] = time.monotonic() + self.interval

    def ready(self):
        with self.condition:
            now = time.monotonic()
            return {
                name
                for name in self.pending | self.deadlines.keys()
                if self.deadlines.get(name, 0) <= now
            }

    def wait(self, should_stop):
        with self.condition:
            while not should_stop() and not self.ready():
                now = time.monotonic()
                delay = min(
                    (max(0, t - now) for t in self.deadlines.values()), default=1.0
                )
                self.condition.wait(min(delay, 1.0))


class Subscriptions:
    """One reader per bus source; no reader touches consumer or lock state."""

    def __init__(self, sources, schedule, log):
        self.stop = threading.Event()
        self.threads = []
        self.schedule = schedule
        self.log = log
        if any(source.is_local for source, _ in sources) and not hasattr(
            select, "kqueue"
        ):
            raise RuntimeError("build notifications require macOS/BSD kqueue")
        for source, engines in sources:
            thread = threading.Thread(
                target=self._run,
                args=(source, engines),
                name=f"bus-{source.name}",
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()

    def _run(self, source, engines):
        delay = 1.0
        while not self.stop.is_set():
            connected = False

            def receive(names):
                nonlocal connected
                connected = True
                self.schedule.changed(names)

            try:
                if source.is_local:
                    subscription = DirectorySubscription(source.local_root, engines)
                    try:
                        receive(engines)
                        while not self.stop.is_set():
                            changed = subscription.changes()
                            if changed:
                                receive(changed)
                    finally:
                        subscription.close()
                else:
                    self._remote(source, engines, receive)
            except (OSError, ValueError, RuntimeError) as exc:
                if not self.stop.is_set():
                    self.log(
                        f"{source.name}: notifications disconnected: {exc}; reconnecting"
                    )
            if connected:
                delay = 1.0
            if self.stop.wait(delay):
                break
            delay = min(delay * 2, 60.0)

    def _remote(self, source, engines, receive):
        from .remote import SSH_OPTIONS, quote_remote

        command = f"slipstream bus subscribe {quote_remote(source.root)}"
        for engine in engines:
            command += f" --engine {shlex.quote(engine)}"
        process = subprocess.Popen(
            ["ssh", *SSH_OPTIONS, source.ssh_host, command],
            stdout=subprocess.PIPE,
            # Inherit stderr: diagnostics cannot fill an unread pipe.
            # Keep the remote endpoint's stdin open until this reader exits.
            stdin=subprocess.PIPE,
        )
        ready = False
        deadline = time.monotonic() + 30
        buffer = b""
        try:
            while not self.stop.is_set():
                if not ready and time.monotonic() >= deadline:
                    raise ValueError("subscription readiness timed out")
                readable, _, _ = select.select([process.stdout], [], [], 1.0)
                if not readable:
                    continue
                data = os.read(process.stdout.fileno(), 65536)
                if not data:
                    raise ValueError("subscription EOF")
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if len(line) > 65536:
                        raise ValueError("subscription message too large")
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("invalid subscription message")
                    kind = message.get("type")
                    names = message.get("engines")
                    if (
                        message.get("version") != PROTOCOL_VERSION
                        or kind not in ("ready", "changed")
                        or not isinstance(names, list)
                        or any(
                            not isinstance(n, str) or n not in engines for n in names
                        )
                        or (not ready and kind != "ready")
                        or (kind == "ready" and (ready or set(names) != set(engines)))
                    ):
                        raise ValueError("invalid subscription protocol")
                    ready = True
                    receive(names)
                if len(buffer) > 65536:
                    raise ValueError("subscription message too large")
        finally:
            process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join()
