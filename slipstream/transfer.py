# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Independent, serial artifact transport. Only the coordinator initiates SSH.

The wire consists of bounded JSON lines and exactly manifest-declared byte
counts. Objects are never decoded or extracted by endpoints. Both directions
use the same capture and acceptance transactions.
"""

from __future__ import annotations

from .service_logging import warning, error, success

import json
import os
import select
import signal
import subprocess
import time
import threading
from contextlib import contextmanager

from .bus import BusError
from .bus_protocol import BusProtocol, encoded, identity, read
from .durability import FileLock, atomic_write
from .models import CommitKey
from .notifications import Subscriptions, WorkSchedule
from .remote import SSH_OPTIONS, quote_remote


TRANSFER_VERSION = 2


@contextmanager
def attempt_deadline(seconds):
    def expired(*_):
        raise TimeoutError("artifact attempt deadline exceeded")

    previous = signal.signal(signal.SIGALRM, expired)
    timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def disconnected_client(fd):
    """Cancel local endpoint work promptly when its control pipe reaches EOF.

    Only used while no protocol input is expected; unexpected input also
    terminates the session. The monitor never takes a bus or machine lock.
    """
    stopped = threading.Event()

    def disconnected(*_):
        raise EOFError("transfer client disconnected")

    previous = signal.signal(signal.SIGUSR1, disconnected)

    def monitor():
        while not stopped.is_set():
            if select.select([fd], [], [], 0.1)[0]:
                if not stopped.is_set():
                    os.kill(os.getpid(), signal.SIGUSR1)
                return

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
        signal.signal(signal.SIGUSR1, previous)


class Wire:
    def __init__(
        self,
        reader,
        writer,
        *,
        handshake=30,
        inactivity=120,
        total=3600,
        should_stop=lambda: False,
    ):
        self.should_stop = should_stop
        self.reader, self.writer = reader, writer
        os.set_blocking(reader, False)
        os.set_blocking(writer, False)
        self.handshake_deadline = time.monotonic() + handshake
        self.deadline = time.monotonic() + total
        self.inactivity = inactivity
        self.handshake = handshake
        self.started = False

    def _wait(self, fd, writing=False):
        deadline = min(
            self.deadline,
            time.monotonic() + self.inactivity
            if self.started
            else self.handshake_deadline,
        )
        while True:
            if self.should_stop():
                raise InterruptedError("transfer cancelled")
            timeout = min(1, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("transfer inactivity or total deadline exceeded")
            ready = select.select(
                [] if writing else [fd], [fd] if writing else [], [], timeout
            )
            if ready[0] or ready[1]:
                return

    def send(self, data):
        view = memoryview(data)
        while view:
            self._wait(self.writer, True)
            try:
                count = os.write(self.writer, view[:65536])
            except BlockingIOError:
                continue
            view = view[count:]

    def receive(self, size):
        self._wait(self.reader)
        data = os.read(self.reader, min(size, 65536))
        if not data:
            raise EOFError("transfer control EOF")
        return data

    def json(self, value):
        self.send((encoded(value) + "\n").encode())

    def message(self):
        data = bytearray()
        while not data.endswith(b"\n"):
            data.extend(self.receive(1))
            if len(data) > 16 * 1024 * 1024:
                raise BusError("transfer message too large")
        value = json.loads(data)
        self.started = True
        if not isinstance(value, dict):
            raise BusError("invalid protocol message")
        if "error" in value:
            if value.get("retryable"):
                raise OSError(value["error"])
            raise BusError(value["error"])
        return value

    def send_objects(self, stage, names):
        objects = {obj.name: obj for obj in stage.entry.objects()}
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) or name not in objects for name in names)
            or len(set(names)) != len(names)
        ):
            raise BusError("invalid missing-object request")
        for name in names:
            obj = objects[name]
            with stage.paths[obj.name].open("rb") as stream:
                while data := stream.read(65536):
                    self.send(data)

    def receive_objects(self, stage):
        for obj in stage.entry.objects():
            if obj.name in stage.paths:
                continue
            path = stage.path / (obj.name + ".partial")
            with path.open("xb") as stream:
                remaining = obj.bytes
                while remaining:
                    data = self.receive(remaining)
                    stream.write(data)
                    remaining -= len(data)
            stage.paths[obj.name] = path


class LocalEndpoint:
    def __init__(self, root, retain_bytes, expected_uuid=None):
        self.protocol = BusProtocol(root, expected_uuid)
        self.retain_bytes = retain_bytes

    def info(self):
        return self.protocol.ready()

    def snapshot(self, engine):
        return self.protocol.snapshot(engine)

    def receipt(self, engine):
        return self.protocol.receipt(engine)

    def skip(self, engine, key, owner):
        return self.protocol.skip(engine, key, owner)

    def health(self, engine):
        result = {}
        for kind in ("builds", "bench"):
            path = self.protocol.state_path(kind, engine)
            if path.exists():
                result[kind] = read(path)
        return result

    def mirror(self, engine, health):
        with self.protocol.lock():
            self.protocol._recover_locked(engine)
            path = self.protocol.state_path("peer-health", engine)
            if path.exists() and read(path).get("source") == health:
                return
            atomic_write(path, encoded({"source": health, "fetched_at": time.time()}))

    @contextmanager
    def capture(self, engine, key, local):
        with self.protocol.capture(engine, key) as stage:
            yield stage

    def accept(self, capture):
        if capture.protocol.root == self.protocol.root:
            # A pull already received directly into destination-owned staging.
            receipt = self.protocol.accept(capture)
            self.protocol.prune(self.retain_bytes)
            self.protocol.sweep_blobs()
            return receipt
        # Link/copy into destination-owned staging before touching its bus.
        with self.protocol.stage() as stage:
            receipt = self.protocol.receipt(capture.entry.engine)
            if receipt.get("identity") == identity(capture.text):
                return receipt
            missing = stage.reuse(capture.text)
            import shutil

            for obj in stage.entry.objects():
                if obj.name not in missing:
                    continue
                dest = stage.path / (obj.name + ".partial")
                shutil.copyfile(capture.paths[obj.name], dest)
                stage.paths[obj.name] = dest
            receipt = self.protocol.accept(stage)
        self.protocol.prune(self.retain_bytes)
        self.protocol.sweep_blobs()
        return receipt


class RemoteEndpoint:
    def __init__(self, endpoint, options, retain_bytes, expected_uuid=None):
        self.expected_uuid = expected_uuid
        self.endpoint, self.options, self.retain_bytes = endpoint, options, retain_bytes

    @contextmanager
    def session(self, request):
        command = (
            'export PATH="$PATH:$HOME/.local/bin"; exec slipstream bus endpoint '
            + quote_remote(self.endpoint.root)
            + " --retain-bytes "
            + str(int(self.retain_bytes))
            + " --deadline "
            + str(self.options["total"])
            + " --inactivity "
            + str(self.options["inactivity"])
            + " --handshake "
            + str(self.options["handshake"])
        )
        process = subprocess.Popen(
            ["ssh", *SSH_OPTIONS, self.endpoint.ssh_host, command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        wire = Wire(process.stdout.fileno(), process.stdin.fileno(), **self.options)
        try:
            wire.json({**request, "expected_uuid": self.expected_uuid})
            yield wire
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            process.stdout.close()

    def call(self, op, **values):
        with self.session({"op": op, **values}) as wire:
            return wire.message()

    def info(self):
        return self.call("info")

    def snapshot(self, engine):
        return self.call("snapshot", engine=engine)

    def receipt(self, engine):
        return self.call("receipt", engine=engine)

    def skip(self, engine, key, owner):
        return self.call("skip", engine=engine, key=str(key), owner=owner)

    def health(self, engine):
        return self.call("health", engine=engine)

    def mirror(self, engine, health):
        self.call("mirror", engine=engine, health=health)

    @contextmanager
    def capture(self, engine, key, local):
        with self.session(
            {
                "op": "capture",
                "engine": engine,
                "key": str(key),
                "transfer_version": TRANSFER_VERSION,
            }
        ) as wire:
            header = wire.message()
            if header.get("transfer_version") != TRANSFER_VERSION:
                raise BusError("transfer protocol changed; update both endpoints")
            with local.stage() as stage:
                wire.json({"missing": stage.reuse(header["manifest"])})
                wire.receive_objects(stage)
                yield stage
                wire.json({"done": True})

    def accept(self, stage):
        with self.session(
            {
                "op": "accept",
                "manifest": stage.text,
                "transfer_version": TRANSFER_VERSION,
            }
        ) as wire:
            header = wire.message()
            if header.get("transfer_version") != TRANSFER_VERSION:
                raise BusError("transfer protocol changed; update both endpoints")
            if "receipt" in header:
                return header["receipt"]
            wire.send_objects(stage, header["missing"])
            return wire.message()


def serve(
    root, reader, writer, retain_bytes, *, deadline=3600, inactivity=120, handshake=30
):
    endpoint = LocalEndpoint(root, retain_bytes)
    wire = Wire(
        reader, writer, total=deadline, inactivity=inactivity, handshake=handshake
    )
    with attempt_deadline(deadline):
        try:
            request = wire.message()
            endpoint.protocol.expected_uuid = request.get("expected_uuid")
            endpoint.info()
            op = request["op"]
            if (
                op in ("accept", "capture")
                and request.get("transfer_version") != TRANSFER_VERSION
            ):
                raise BusError("transfer protocol changed; update both endpoints")
            if op == "accept":
                ident = identity(request["manifest"])
                receipt = endpoint.receipt(ident["engine"])
                if receipt.get("identity") == ident:
                    wire.json(
                        {"transfer_version": TRANSFER_VERSION, "receipt": receipt}
                    )
                    return
                with endpoint.protocol.stage() as stage:
                    missing = stage.reuse(request["manifest"])
                    wire.json(
                        {"transfer_version": TRANSFER_VERSION, "missing": missing}
                    )
                    wire.receive_objects(stage)
                    with disconnected_client(reader):
                        receipt = endpoint.protocol.accept(stage)
                        endpoint.protocol.prune(retain_bytes)
                        endpoint.protocol.sweep_blobs()
                wire.json(receipt)
            elif op == "capture":
                with endpoint.protocol.capture(
                    request["engine"], request["key"]
                ) as stage:
                    wire.json(
                        {"transfer_version": TRANSFER_VERSION, "manifest": stage.text}
                    )
                    wire.send_objects(stage, wire.message()["missing"])
                    wire.message()
            elif op in ("snapshot", "receipt", "health"):
                wire.json(getattr(endpoint, op)(request["engine"]))
            elif op == "info":
                wire.json(endpoint.info())
            elif op == "skip":
                wire.json(
                    endpoint.skip(request["engine"], request["key"], request["owner"])
                )
            elif op == "mirror":
                endpoint.mirror(request["engine"], request["health"])
                wire.json({"ok": True})
            else:
                raise BusError("unknown endpoint operation")
        except EOFError:
            return
        except (BusError, OSError, ValueError, KeyError) as exc:
            wire.json({"error": str(exc), "retryable": isinstance(exc, OSError)})


class Coordinator:
    def __init__(self, cfg, log=print):
        self.cfg, self.log = cfg, log
        self.local = BusProtocol(cfg.bus.root)
        ready = self.local.ready()
        self.local.expected_uuid = ready["uuid"]
        if cfg.bus.owners and ready["owners"] != cfg.bus.owners:
            raise BusError("configured ownership differs from migrated bus")
        self.routes = {}
        self.bounds = {}
        self.failed = set()
        self.health_generation = {}
        self.options = {
            "handshake": cfg.bus.transfer_handshake,
            "inactivity": cfg.bus.transfer_inactivity,
            "total": cfg.bus.transfer_deadline,
        }
        for route in cfg.bus.routes:

            def endpoint(spec, expected_uuid):
                return (
                    LocalEndpoint(
                        spec.local_root, cfg.build.retain_gb * 1e9, expected_uuid
                    )
                    if spec.is_local
                    else RemoteEndpoint(
                        spec, self.options, cfg.build.retain_gb * 1e9, expected_uuid
                    )
                )

            source, dest = (
                endpoint(route.source, route.source_uuid),
                endpoint(route.destination, route.destination_uuid),
            )
            for engine in route.engines:
                self.routes[engine] = (route, source, dest)

    def path(self, engine):
        return self.local.state_path("transfer", engine)

    def save(self, engine, record):
        with self.local.lock():
            self.local.ready()
            atomic_write(self.path(engine), encoded(record))

    def initialize(self, engine, *, once):
        route, source, dest = self.routes[engine]
        src, dst = source.info(), dest.info()
        if (
            src["uuid"] == dst["uuid"]
            or src["owners"].get(engine) != src["uuid"]
            or dst["owners"].get(engine) != src["uuid"]
        ):
            raise BusError("route UUID or configured ownership conflict")
        path = self.path(engine)
        if not path.exists():
            raise BusError(
                "missing coordinator state; run bus migrate with route configuration"
            )
        record = read(path)
        if record["source"] != src["uuid"] or record["destination"] != dst["uuid"]:
            raise BusError("endpoint UUID changed; reconcile transfer progress offline")
        snapshot = source.snapshot(engine)
        if once:
            values = snapshot["keys"] + (
                [snapshot["retention"]["highest_dropped"]]
                if snapshot["retention"]["highest_dropped"]
                else []
            )
            self.bounds[engine] = max(map(CommitKey.parse, values), default=None)
        cursor = CommitKey.parse(record["cursor"]) if record["cursor"] else None
        remaining = sum(
            cursor is None or CommitKey.parse(key) > cursor for key in snapshot["keys"]
        )
        bound = f", through={self.bounds[engine]}" if once else ""
        self.log(
            f"{engine}: route={route.name}, cursor={record['cursor'] or 'none'}, "
            f"queued={remaining}{bound}"
        )
        return record

    def _finish(self, engine, record, reply):
        pending = record["pending"]
        key = CommitKey.parse(pending["key"])
        if reply.get("identity"):
            if reply["identity"] != pending.get("identity"):
                raise BusError("destination receipt conflicts with pending publication")
            record["acknowledged"] = reply["identity"]
        elif reply.get("state") == "skipped" and reply["key"] == str(key):
            record["highest_skipped"] = str(key)
        else:
            raise BusError("destination watermark unexpectedly ahead or conflicting")
        if pending["op"] == "deliver":
            record["backlog"] = max(0, record.get("backlog", 1) - 1)
        record.update(
            cursor=str(key), pending=None, retry_after=0, failures=0, last_error=None
        )
        self.save(engine, record)
        if pending["op"] == "skip":
            self.log(warning(f"{engine}: skipped unavailable history through {key}"))

    def step(self, engine, record, *, once=False):
        route, source, dest = self.routes[engine]
        pending = record.get("pending")
        if pending:
            receipt = dest.receipt(engine)
            boundary = receipt.get("key") or receipt.get("identity", {}).get("key")
            if boundary and CommitKey.parse(boundary) >= CommitKey.parse(
                pending["key"]
            ):
                self._finish(engine, record, receipt)
                self.log(
                    success(
                        f"{engine}: recovered acknowledgement through {record['cursor']}"
                    )
                )
                return True
            if pending["op"] == "skip":
                self._finish(
                    engine, record, dest.skip(engine, pending["key"], record["source"])
                )
                return True
        snapshot = source.snapshot(engine)
        cursor = CommitKey.parse(record["cursor"]) if record["cursor"] else None
        keys = [
            CommitKey.parse(k)
            for k in snapshot["keys"]
            if cursor is None or CommitKey.parse(k) > cursor
        ]
        if once:
            bound = self.bounds[engine]
            keys = [k for k in keys if bound is not None and k <= bound]
        record["backlog"] = len(keys)
        dropped = snapshot["retention"]["highest_dropped"]
        dropped = CommitKey.parse(dropped) if dropped else None
        if once and dropped and self.bounds[engine] and dropped > self.bounds[engine]:
            dropped = self.bounds[engine]
        target = (
            CommitKey.parse(pending["key"]) if pending else (keys[0] if keys else None)
        )
        # Only fence missing history below the first retained entry.
        if pending and target not in keys:
            if dropped is None or dropped < target or any(k < target for k in keys):
                raise BusError("missing pending publication without retention evidence")
            record["pending"] = {
                **pending,
                "op": "skip",
                "evidence": snapshot["retention"],
            }
            self.save(engine, record)
            self._finish(engine, record, dest.skip(engine, target, record["source"]))
            return True
        if (
            not pending
            and dropped
            and (cursor is None or dropped > cursor)
            and (not keys or dropped < keys[0])
        ):
            record["pending"] = {
                "op": "skip",
                "key": str(dropped),
                "evidence": snapshot["retention"],
            }
            self.save(engine, record)
            self._finish(engine, record, dest.skip(engine, dropped, record["source"]))
            return True
        if target is None:
            self.save(engine, record)
            return False
        started = time.monotonic()
        with source.capture(engine, target, self.local) as stage:
            ident = identity(stage.text)
            if pending and pending["identity"] != ident:
                raise BusError("source publication changed while delivery pending")
            record["pending"] = {"op": "deliver", "key": str(target), "identity": ident}
            self.save(engine, record)
            self._finish(engine, record, dest.accept(stage))
        self.log(
            success(
                f"{engine}: accepted {target}, remaining={record['backlog']}, "
                f"elapsed={time.monotonic() - started:.2f}s"
            )
        )
        return True

    def copy_health(self, engine, record, generation):
        if (
            record.get("health_retry_after", 0) > time.time()
            and self.health_generation.get(engine) == generation
        ):
            return
        self.health_generation[engine] = generation
        _, source, dest = self.routes[engine]
        try:
            dest.mirror(engine, source.health(engine))
        except (OSError, EOFError, BusError, ValueError) as exc:
            record["health_error"] = str(exc)
            record["health_failures"] = record.get("health_failures", 0) + 1
            record["health_retry_after"] = time.time() + min(
                60, 2 ** min(record["health_failures"] - 1, 6)
            )
            self.log(warning(f"{engine}: health snapshot: {exc}"))
            self.save(engine, record)
        else:
            if record.get("health_error"):
                record.update(
                    health_error=None, health_retry_after=0, health_failures=0
                )
                self.save(engine, record)

    @staticmethod
    def retry_delay(record, artifact=True):
        fields = (
            ("retry_after", "health_retry_after")
            if artifact
            else ("health_retry_after",)
        )
        delays = [
            max(0, record.get(field, 0) - time.time())
            for field in fields
            if record.get(field, 0) > time.time()
        ]
        return min(60, min(delays)) if delays else None

    def run(self, *, once=False, should_stop=lambda: False):
        with FileLock(self.local.root / ".coordinator.lock", timeout=0):
            self.options["should_stop"] = should_stop
            sources = {}
            for engine, (route, _, _) in self.routes.items():
                key = (route.source.root, route.source.ssh_host)
                sources.setdefault(key, (route.source, []))[1].append(engine)
            schedule = WorkSchedule(list(self.routes), [], 0)
            schedule.wake_on_change = True  # health events bypass artifact backoff
            subscriptions = (
                None
                if once
                else Subscriptions(
                    list(sources.values()), schedule, self.log, role="transfer"
                )
            )
            records, initialized = {}, set()
            try:
                for engine in self.routes:
                    try:
                        records[engine] = read(self.path(engine))
                        if records[engine].get("conflict"):
                            self.failed.add(engine)
                    except BusError as exc:
                        self.log(error(f"{engine}: {exc}"))
                        self.failed.add(engine)
                # --once freezes every upper bound before delivering any work.
                if once:
                    for engine in records:
                        try:
                            self.initialize(engine, once=True)
                            initialized.add(engine)
                        except (OSError, BusError, EOFError) as exc:
                            record = records[engine]
                            record["last_error"] = str(exc)
                            record["retry_after"] = time.time() + 1
                            if isinstance(exc, BusError):
                                record["conflict"] = True
                            self.save(engine, record)
                            self.log(error(f"{engine}: {exc}"))
                            self.failed.add(engine)
                schedule.changed(records)
                while not should_stop():
                    progressed = False
                    for engine, record in records.items():
                        generation = schedule.snapshot(engine)
                        if engine in self.failed:
                            schedule.finished(engine, generation)
                            continue
                        if not once and engine not in schedule.ready():
                            continue
                        try:
                            with attempt_deadline(self.options["total"]):
                                if engine not in initialized:
                                    self.initialize(engine, once=False)
                                    initialized.add(engine)
                                self.copy_health(engine, record, generation)
                                if record["retry_after"] > time.time():
                                    schedule.finished(
                                        engine,
                                        generation,
                                        retry=self.retry_delay(record),
                                    )
                                    continue
                                more = self.step(engine, record, once=once)
                            progressed |= more
                            schedule.finished(
                                engine,
                                generation,
                                more=more,
                                retry=None
                                if more
                                else self.retry_delay(record, artifact=False),
                            )
                        except (OSError, EOFError, BusError, ValueError) as exc:
                            record["last_error"] = str(exc)
                            record["failures"] = record.get("failures", 0) + 1
                            delay = min(60, 2 ** min(record["failures"] - 1, 6))
                            record["retry_after"] = time.time() + delay
                            if isinstance(exc, BusError):
                                record["conflict"] = True
                                self.failed.add(engine)
                            self.save(engine, record)
                            self.log(error(f"{engine}: {exc}"))
                            schedule.finished(
                                engine,
                                generation,
                                retry=None
                                if engine in self.failed
                                else self.retry_delay(record),
                            )
                    if once and not progressed:
                        return not self.failed and all(
                            not r["pending"]
                            and (
                                self.bounds[e] is None
                                or (
                                    r["cursor"]
                                    and CommitKey.parse(r["cursor"]) >= self.bounds[e]
                                )
                            )
                            for e, r in records.items()
                        )
                    if not once:
                        schedule.wait(should_stop)
                return False
            finally:
                if subscriptions:
                    subscriptions.close()
