# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Version 4 bus transactions. All *_locked methods require the root lock.

Recovery order is acceptance (prepared -> complete, skipped -> loss summary),
then pending prune. Complete receipts never resurrect retained-away entries.
Private staging holds an independent flock; cleanup only probes it nonblocking.
No hashing, network I/O or machine-lock acquisition occurs inside transactions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import replace
from contextvars import ContextVar
from pathlib import Path

from .bus import Bus, BusError, Entry, sha256_file, state_to_json
from .durability import FileLock, atomic_write, durable_mkdir, flush_file, fsync_dir
from .models import CommitKey


_active_staging = ContextVar("bus_staging_fds", default=())


def staging_fds():
    """Pass to children that use staging; a parent crash must not release it."""
    return _active_staging.get()


def sync_metadata(path):
    # Re-sync visible state after an earlier sync error; visibility alone does
    # not prove the previous attempt reached stable storage.
    with path.open("rb") as stream:
        flush_file(stream.fileno())
        fsync_dir(path.parent)
        flush_file(stream.fileno())


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(data):
    return hashlib.sha256(data.encode()).hexdigest()


def manifest_text(path):
    # Preserve persisted UTF-8 bytes, including whitespace/newline spelling.
    # Path.read_text() would normalize CRLF before computing transfer identity.
    return path.read_bytes().decode("utf-8")


def read(path):
    try:
        value = json.loads(path.read_text())
    except (ValueError, FileNotFoundError) as exc:
        raise BusError(f"missing or invalid operational state: {path}") from exc
    if not isinstance(value, dict):
        raise BusError(f"invalid operational state: {path}")
    return value


def component(value):
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
    ):
        raise BusError(f"invalid engine: {value!r}")
    return value


def identity(text):
    entry = Entry.from_json(text)
    if entry.version not in (4, 5):
        raise BusError("transfer requires offline migration to version 4")
    component(entry.engine)
    try:
        uuid.UUID(entry.owner_uuid)
        uuid.UUID(entry.publication_uuid)
    except (ValueError, AttributeError, TypeError) as exc:
        raise BusError("invalid publication UUID") from exc
    for obj in entry.objects():
        if (
            len(obj.sha256) != 64
            or any(c not in "0123456789abcdef" for c in obj.sha256)
            or obj.bytes < 0
        ):
            raise BusError("invalid object identity")
    return {
        "owner": entry.owner_uuid,
        "engine": entry.engine,
        "key": str(entry.key),
        "publication": entry.publication_uuid,
        "sha256": digest(text),
    }


class Staging:
    def __init__(self, protocol, path, owner):
        self.protocol, self.path, self.owner = protocol, path, owner
        self.text = None
        self.entry = None
        self.paths = {}
        self.reused = {}

    def reuse(self, text):
        """Pin immutable, previously verified objects before negotiating bytes."""
        identity(text)
        self.text, self.entry = text, Entry.from_json(text)
        with self.protocol.lock():
            ready = self.protocol.ready()
            if ready["owners"].get(self.entry.engine) != self.entry.owner_uuid:
                raise BusError("source is not configured owner")
            for obj in self.entry.objects():
                path = self.path / obj.name
                try:
                    os.link(self.protocol.bus.object_path(obj.name), path)
                except FileNotFoundError:
                    continue
                if path.stat().st_size != obj.bytes:
                    raise BusError(f"existing object size mismatch: {obj.name}")
                self.paths[obj.name] = path
                self.reused[obj.name] = path
        return [o.name for o in self.entry.objects() if o.name not in self.paths]

    def verify(self):
        for obj in self.entry.objects():
            path = self.paths[obj.name]
            if obj.name in self.reused:
                continue
            if path.stat().st_size != obj.bytes or sha256_file(path) != obj.sha256:
                raise BusError(f"object checksum mismatch: {obj.name}")
            with path.open("rb") as stream:
                flush_file(stream.fileno())

    def close(self):
        if self.owner is not None:
            try:
                with self.protocol.lock():
                    shutil.rmtree(self.path)
            finally:
                self.owner.close()
                self.owner = None

    def __enter__(self):
        self._token = _active_staging.set((*staging_fds(), self.owner._file.fileno()))
        return self

    def __exit__(self, *args):
        try:
            self.close()
        finally:
            _active_staging.reset(self._token)


class BusProtocol:
    def __init__(self, root, expected_uuid=None):
        self.expected_uuid = expected_uuid
        self.root = Path(root).expanduser().resolve()
        self.bus = Bus(self.root, _raw=True)

    def lock(self):
        return FileLock(self.root / ".bus.lock")

    def state_path(self, kind, engine):
        return self.root / "state" / kind / f"{component(engine)}.json"

    def ready(self):
        data = read(self.root / "ready.json")
        if self.expected_uuid is not None and data.get("uuid") != self.expected_uuid:
            raise BusError("endpoint UUID changed; reconcile progress offline")
        if data.get("schema") != 4:
            raise BusError("unsupported bus readiness schema")
        if read(self.root / "identity.json")["uuid"] != data["uuid"]:
            raise BusError("bus UUID differs from readiness marker")
        return data

    def bus_uuid(self):
        with self.lock():
            path = self.root / "identity.json"
            if not path.exists():
                if (self.root / "ready.json").exists():
                    raise BusError("initialized bus lost identity")
                atomic_write(path, encoded({"uuid": str(uuid.uuid4())}))
            state = read(path)
            if state.get("initialized") and not (self.root / "ready.json").exists():
                raise BusError(
                    "initialized bus lost readiness marker; restore it, do not reset"
                )
            return state["uuid"]

    def _stage_locked(self):
        path = self.root / "staging" / str(uuid.uuid4())
        durable_mkdir(path)
        owner = FileLock(path / ".owner", timeout=0)
        owner.__enter__()
        return Staging(self, path, owner)

    def stage(self):
        with self.lock():
            return self._stage_locked()

    def _capture_locked(self, engine, key):
        text = manifest_text(self.bus.entry_path(engine, key))
        entry = Entry.from_json(text)
        if entry.engine != engine or entry.key != CommitKey.of(key):
            raise BusError("manifest does not match its topic/key")
        stage = self._stage_locked()
        try:
            stage.text, stage.entry = text, entry
            for obj in entry.objects():
                path = stage.path / obj.name
                os.link(self.bus.entry_object_path(entry, obj), path)
                stage.paths[obj.name] = path
        except BaseException:
            shutil.rmtree(stage.path)
            stage.owner.close()
            raise
        return stage

    def capture(self, engine, key):
        with self.lock():
            self._recover_locked(engine)
            return self._capture_locked(engine, key)

    def _write(self, kind, engine, data):
        path = self.state_path(kind, engine)
        text = encoded(data)
        if path.exists() and path.read_text() == text:
            sync_metadata(path)
            return
        atomic_write(path, text)

    def _recover_locked(self, engine):
        try:
            self._recover_state_locked(engine)
        except (KeyError, TypeError, ValueError) as exc:
            raise BusError(f"malformed operational state for {engine}: {exc}") from exc

    def _recover_state_locked(self, engine):
        ready = self.ready()
        if engine not in ready["owners"]:
            raise BusError(f"no configured owner for {engine}")
        journal = read(self.state_path("acceptance", engine))
        state = journal.get("state")
        if state in ("prepared", "complete"):
            if not {"manifest", "identity"} <= journal.keys():
                raise BusError("incomplete acceptance record")
            ident = identity(journal["manifest"])
            if (
                ident != journal["identity"]
                or ident["engine"] != engine
                or ident["owner"] != ready["owners"][engine]
            ):
                raise BusError(
                    "acceptance identity conflicts with configured engine/owner"
                )
        if state == "skipped":
            if journal.get("owner") != ready["owners"][engine]:
                raise BusError("skip ownership conflict")
            try:
                CommitKey.parse(journal["key"])
            except (KeyError, ValueError, TypeError) as exc:
                raise BusError("invalid skip boundary") from exc
        if state == "prepared":
            text = journal["manifest"]
            ident = identity(text)
            if (
                ident != journal["identity"]
                or ident["owner"] != ready["owners"][engine]
            ):
                raise BusError("corrupt prepared acceptance identity")
            entry = Entry.from_json(text)
            for obj in entry.objects():
                if not self.bus.object_path(obj.name).is_file():
                    raise BusError(f"prepared journal is missing {obj.name}")
            self._manifest_locked(entry, text)
            journal["state"] = "complete"
            self._write("acceptance", engine, journal)
        elif state == "skipped":
            self._write("upstream-loss", engine, {"highest_dropped": journal["key"]})
        elif state not in ("empty", "complete"):
            raise BusError(f"invalid acceptance journal: {engine}")
        retention = read(self.state_path("retention", engine))
        if not {"pending", "highest_dropped", "lowest_retained"} <= retention.keys():
            raise BusError("invalid retention record")
        pending = retention["pending"]
        if not isinstance(pending, list) or len(pending) > 256:
            raise BusError("invalid pending prune batch")
        if pending:
            for item in pending:
                path = self.bus.entry_path(engine, item["key"])
                if path.exists() and digest(manifest_text(path)) != item["sha256"]:
                    raise BusError(
                        "pending prune conflicts with replacement publication"
                    )
                path.unlink(missing_ok=True)
            fsync_dir(self.bus.topic_dir(engine))
            keys = self.bus.keys(engine)
            old = retention.get("highest_dropped")
            highest = max(
                [CommitKey.parse(i["key"]) for i in pending]
                + ([CommitKey.parse(old)] if old else [])
            )
            retention.update(
                highest_dropped=str(highest),
                lowest_retained=str(keys[0]) if keys else None,
                pending=[],
            )
            self._write("retention", engine, retention)
        self._refresh_retention_locked(engine)
        sync_metadata(self.state_path("acceptance", engine))
        sync_metadata(self.state_path("retention", engine))

    def _refresh_retention_locked(self, engine):
        state = read(self.state_path("retention", engine))
        keys = self.bus.keys(engine)
        lowest = str(keys[0]) if keys else None
        if state["lowest_retained"] != lowest:
            state["lowest_retained"] = lowest
            self._write("retention", engine, state)

    def _recover_all_locked(self):
        for engine in self.ready()["owners"]:
            self._recover_locked(engine)

    def _manifest_locked(self, entry, text):
        path = self.bus.entry_path(entry.engine, entry.key)
        if path.exists() and manifest_text(path) != text:
            raise BusError(f"publication conflict: {entry.engine} {entry.key}")
        atomic_write(path, text)

    def keys(self, engine):
        with self.lock():
            self._recover_locked(engine)
            return self.bus.keys(engine)

    def read_entry(self, engine, key):
        with self.lock():
            self._recover_locked(engine)
            return self.bus.read_entry(engine, key)

    def snapshot(self, engine):
        with self.lock():
            self._recover_locked(engine)
            return {
                "uuid": self.ready()["uuid"],
                "keys": [str(k) for k in self.bus.keys(engine)],
                "retention": read(self.state_path("retention", engine)),
            }

    def receipt(self, engine):
        with self.lock():
            self._recover_locked(engine)
            return read(self.state_path("acceptance", engine))

    def skip(self, engine, key, owner):
        key = CommitKey.of(key)
        with self.lock():
            self._recover_locked(engine)
            if self.ready()["owners"][engine] != owner:
                raise BusError("source is not configured owner")
            journal = read(self.state_path("acceptance", engine))
            if journal["state"] != "empty":
                boundary = CommitKey.parse(
                    journal.get("key") or journal["identity"]["key"]
                )
                if boundary >= key:
                    return journal
            journal = {"state": "skipped", "key": str(key), "owner": owner}
            self._write("acceptance", engine, journal)
            self._recover_locked(engine)
            return journal

    def _install_locked(self, stage):
        durable_mkdir(self.bus.blobs_dir)
        for obj in stage.entry.objects():
            dest = self.bus.object_path(obj.name)
            try:
                os.link(stage.paths[obj.name], dest)
            except FileExistsError:
                # Existing names were captured and verified before entering.
                prior = stage.existing.get(obj.name)
                if prior is None or not os.path.samestat(prior.stat(), dest.stat()):
                    raise OSError(
                        f"object appeared during verification; retry: {obj.name}"
                    )
        fsync_dir(self.bus.blobs_dir)

    def verify_install(self, stage):
        stage.existing = {}
        with self.lock():
            for obj in stage.entry.objects():
                path = stage.path / (str(uuid.uuid4()) + "-" + obj.name)
                try:
                    os.link(self.bus.object_path(obj.name), path)
                except FileNotFoundError:
                    continue
                stage.existing[obj.name] = path
        stage.verify()
        for obj in stage.entry.objects():
            path = stage.existing.get(obj.name)
            reused = stage.reused.get(obj.name)
            if path and reused and os.path.samestat(path.stat(), reused.stat()):
                continue
            if path and (
                path.stat().st_size != obj.bytes or sha256_file(path) != obj.sha256
            ):
                raise BusError(f"corrupt existing object: {obj.name}")

    def accept(self, stage):
        ident = identity(stage.text)
        entry = Entry.from_json(stage.text)
        if entry != stage.entry:
            raise BusError("staged manifest differs from object plan")
        self.verify_install(stage)
        with self.lock():
            self._recover_locked(entry.engine)
            if ident["owner"] != self.ready()["owners"][entry.engine]:
                raise BusError("source is not configured owner")
            journal = read(self.state_path("acceptance", entry.engine))
            if journal["state"] != "empty":
                if journal.get("identity") == ident:
                    return journal
                boundary = CommitKey.parse(
                    journal.get("key") or journal["identity"]["key"]
                )
                if entry.key <= boundary:
                    raise BusError("publication conflicts with destination watermark")
            for previous in self.bus.entries(entry.engine):
                if (
                    previous.publication_uuid == entry.publication_uuid
                    and previous.to_json() != entry.to_json()
                ):
                    raise BusError("publication UUID reused with different bytes")
            if journal.get("identity", {}).get("publication") == ident["publication"]:
                raise BusError("publication UUID reused with different bytes")
            path = self.bus.entry_path(entry.engine, entry.key)
            if path.exists() and manifest_text(path) != stage.text:
                raise BusError("conflicting existing publication")
            self._install_locked(stage)
            journal = {"state": "prepared", "identity": ident, "manifest": stage.text}
            self._write("acceptance", entry.engine, journal)
            self._recover_locked(entry.engine)
            return read(self.state_path("acceptance", entry.engine))

    def publish(self, entry):
        raise BusError("version 4 publication requires a protected staging transaction")

    def publish_staged(self, stage):
        ident = identity(stage.text)
        if Entry.from_json(stage.text) != stage.entry:
            raise BusError("staged manifest differs from object plan")
        self.verify_install(stage)
        with self.lock():
            self._recover_locked(stage.entry.engine)
            ready = self.ready()
            if (
                ident["owner"] != ready["uuid"]
                or ready["owners"][stage.entry.engine] != ready["uuid"]
            ):
                raise BusError("builder does not own this engine")
            self._install_locked(stage)
            self._manifest_locked(stage.entry, stage.text)
            self._refresh_retention_locked(stage.entry.engine)

    def write_builder_state(self, engine, state):
        with self.lock():
            self._recover_locked(engine)
            if self.ready()["owners"][engine] != self.ready()["uuid"]:
                raise BusError(
                    "only the configured builder may write original build state"
                )
            state.updated_at = time.time()
            atomic_write(self.bus.builder_state_path(engine), state_to_json(state))

    def write_bench_state(self, engine, state):
        with self.lock():
            self._recover_locked(engine)
            state.updated_at = time.time()
            atomic_write(self.bus.bench_state_path(engine), state_to_json(state))

    def prune(self, retain_bytes, keep=None):
        # Select with the existing shared-object budget algorithm on a metadata
        # snapshot; its unlink calls are redirected to collecting candidates.
        with self.lock():
            self._recover_all_locked()
            candidates = {}

            class SelectionBus(Bus):
                def _delete_pruned(inner, engine, key):
                    candidates.setdefault(engine, []).append(CommitKey.of(key))

            SelectionBus(self.root, _raw=True).prune(retain_bytes, keep)
            for engine, keys in candidates.items():
                for start in range(0, len(keys), 256):
                    state = read(self.state_path("retention", engine))
                    state["pending"] = [
                        {
                            "key": str(k),
                            "sha256": digest(
                                manifest_text(self.bus.entry_path(engine, k))
                            ),
                        }
                        for k in keys[start : start + 256]
                    ]
                    self._write("retention", engine, state)
                    self._recover_locked(engine)
            return candidates

    def sweep_blobs(self):
        import logging

        waited = time.monotonic()
        with self.lock():
            started = time.monotonic()
            self._recover_all_locked()
            result = self.bus.sweep_blobs()
            self.last_sweep_seconds = time.monotonic() - started
        logging.getLogger(__name__).info(
            "bus sweep wait=%.6fs hold=%.6fs removed=%d",
            started - waited,
            self.last_sweep_seconds,
            len(result),
        )
        return result

    def gc(self):
        removed = self.sweep_blobs()
        with self.lock():
            for path in (self.root / "staging").glob("*"):
                owner = FileLock(path / ".owner", timeout=0)
                try:
                    owner.__enter__()
                except TimeoutError:
                    continue
                try:
                    shutil.rmtree(path)
                    removed.append(path)
                finally:
                    owner.close()
        # The command caller holds the machine lock for recursive run cleanup.
        for path in (self.root / "roots").glob("*/*.unpacking"):
            shutil.rmtree(path)
            removed.append(path)
        return removed

    def migrate(self, owners, routes=()):
        """Offline only. Resume per entry, keeping old names until all convert."""
        local_uuid = self.bus_uuid()
        owners = {component(k): str(uuid.UUID(v)) for k, v in owners.items()}
        with self.lock():
            if (self.root / "ready.json").exists():
                if self.ready()["owners"] != owners:
                    raise BusError("ownership change requires offline reconciliation")
                self._recover_all_locked()
                for route in routes:
                    for engine in route.engines:
                        progress = read(self.state_path("transfer", engine))
                        if (
                            progress["source"] != route.source_uuid
                            or progress["destination"] != route.destination_uuid
                        ):
                            raise BusError(
                                "route change requires offline reconciliation"
                            )
                atomic_write(
                    self.root / "identity.json",
                    encoded({"uuid": local_uuid, "initialized": True}),
                )
                return []
            if set(self.bus.engines()) - owners.keys():
                raise BusError("every existing engine needs a configured owner")
        migrated = []
        for engine in owners:
            for key in self.bus.keys(engine):
                with self.lock():
                    capture = self._capture_locked(engine, key)
                with capture:
                    capture.verify()
                    old = capture.entry
                    if old.version >= 4:
                        identity(capture.text)
                        if old.owner_uuid != owners[engine]:
                            raise BusError("migration owner conflict")
                        continue
                    entry = replace(
                        old, version=4, owner_uuid=owners[engine], publication_uuid=""
                    )
                    canonical = json.loads(entry.to_json())
                    del canonical["publication_uuid"]
                    entry.publication_uuid = str(
                        uuid.uuid5(uuid.UUID(entry.owner_uuid), encoded(canonical))
                    )
                    paths = {}
                    for blob in old.blobs:
                        for before, after in zip(
                            blob.objects(old.version), blob.objects(4)
                        ):
                            paths[after.name] = capture.paths[before.name]
                    capture.paths, capture.entry, capture.text = (
                        paths,
                        entry,
                        entry.to_json(),
                    )
                    self.verify_install(capture)
                    with self.lock():
                        self._install_locked(capture)
                        atomic_write(self.bus.entry_path(engine, key), capture.text)
                    migrated.append(key)
        with self.lock():
            for engine in owners:
                state = self.bus.read_builder_state(engine)
                keys = self.bus.keys(engine)
                self._write("acceptance", engine, {"state": "empty"})
                self._write(
                    "retention",
                    engine,
                    {
                        "highest_dropped": str(state.highest_dropped)
                        if state.highest_dropped
                        else None,
                        "lowest_retained": str(keys[0]) if keys else None,
                        "pending": [],
                    },
                )
                self._write("upstream-loss", engine, {"highest_dropped": None})
                bench = self.bus.read_bench_state(engine)
                bench.blobs = []
                bench.updated_at = time.time()
                atomic_write(self.bus.bench_state_path(engine), state_to_json(bench))
            for path in self.bus.tmp_dir.glob("*"):
                if path.is_file():
                    path.unlink()
            legacy = self.root / "blobs" / "builds"
            if legacy.exists():
                shutil.rmtree(legacy)
                fsync_dir(self.bus.blobs_dir)
            if self.bus.tmp_dir.exists():
                fsync_dir(self.bus.tmp_dir)
            for route in routes:
                for engine in route.engines:
                    self._write(
                        "transfer",
                        engine,
                        {
                            "source": route.source_uuid,
                            "destination": route.destination_uuid,
                            "cursor": None,
                            "acknowledged": None,
                            "highest_skipped": None,
                            "pending": None,
                            "retry_after": 0,
                            "failures": 0,
                            "last_error": None,
                            "conflict": False,
                        },
                    )
            atomic_write(
                self.root / "ready.json",
                encoded({"schema": 4, "uuid": local_uuid, "owners": owners}),
            )
            atomic_write(
                self.root / "identity.json",
                encoded({"uuid": local_uuid, "initialized": True}),
            )
        return migrated
