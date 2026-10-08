# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""The build bus: one box builds a commit, both boxes bench the artifact.

Layout, one root per machine::

    bus/topics/builds/<engine>/<key>.json    entries: manifests naming blobs
    bus/blobs/<id>.tar.zst                   blobs, flat, shared by every engine
    bus/state/builds/<engine>.json           builder state, published
    bus/state/bench/<engine>.json            local bencher state
    bus/roots/<engine>/<key>/                unpacked union of an entry's blobs
    bus/tmp/                                 in-flight packaging and fetches

Entries are keyed by commit key, not by a sequence number: the payload already
carries a monotone, machine-independent key. A cursor is therefore a key and a
consumer takes the next entry above it, so nothing needs contiguity and there
is no gap detection or seq allocation. The builder publishes strictly upward
per engine, because a key cursor cannot see anything published at or below
it; backfilling an older range needs an explicit cursor reset on every
consumer.

A key is ``str(CommitKey)``: the bare commit id for an engine that is its
own embedder, ``<embedder_id>-<commit_id>`` otherwise. Files written before
keys had two parts are therefore already named correctly.

An entry is a manifest. Each entry of the engine's ``run_set`` is one blob,
named by the hash of its tree, so a run set entry a commit did not change --
ICU for every V8 commit, most of the WebKit runtime for a JSC-only commit, a
browser's host application until it is updated -- is archived once and
named by every manifest that needs it. Dedupe on publish, the consumer's
fetch cache and retention are then the same thing: reference counting over
manifests. Nothing declares what is shared; the run set already says where
the boundaries are, and it is already what ``build_cfg_hash`` covers.

Entries written as version 1 named one payload per entry under
``blobs/builds/<engine>/<key>.tar.zst``. They read back as a manifest with a
single blob whose id is its sha256, found at the old path until the builder
has migrated it (``Bus.migrate``), which is a hardlink and a rewrite.

Version 3 adds the delta shape of a blob: the same tree, stored as a base
archive plus one bsdiff patch per changed 8 MiB block of the tar stream
(``delta.py``). Patches are files beside the archives, ``blobs/<sha>.bsdiff``,
named by their own sha256. A blob therefore maps to one or more *stored
objects* -- its archive, or its base's archive and its patches -- and
everything below the manifest (fetch, retention, the sweep, the bencher's
pin) iterates objects, not blobs. Only provisioning tells the shapes apart.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path

from .delta import DeltaError, DeltaPlan
from .models import CommitKey

# State files and cursors. Their schema has not changed.
VERSION = 1
# Entries. Versions 1 and 2 are read, version 3 is written.
ENTRY_VERSION = 3
READABLE_ENTRY_VERSIONS = (1, 2, 3)

BLOB_SUFFIX = ".tar.zst"
PATCH_SUFFIX = ".bsdiff"
OBJECT_SUFFIXES = (BLOB_SUFFIX, PATCH_SUFFIX)


def parse_key_stem(stem: str) -> CommitKey | None:
    """A filename stem as a key, or None for anything that is not one.

    Tmp files and strays share the directory; a stem that is not a key is
    simply not an entry, not an error.
    """
    try:
        return CommitKey.parse(stem)
    except ValueError:
        return None


class BusError(RuntimeError):
    """The bus root holds something this version cannot read."""


def _atomic_write(path: Path, data: str) -> None:
    """Write via a uniquely named tmp file and rename.

    Unique because two writers exist per machine (the daemon and an ad-hoc
    command), and rename rather than in-place because box1 reads these files
    over ssh, where a torn read would surface as a parse error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns():x}")
    tmp.write_text(data)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(src_dir: Path, relpath: str) -> str:
    """The identity of one run set entry: what unpacking its blob produces.

    A sha256 over the entry's path and, for every file beneath it in sorted
    order, its own path, kind, executable bit, symlink target and bytes. The
    path is in because the blob unpacks to it: the same bytes at another
    place are a different result. Mtimes and ownership are out because tar
    records them and they differ between two builds of identical output,
    which is exactly the case this exists to recognise. Symlinks are hashed
    as links, not followed, which is also how tar archives them.
    """
    h = hashlib.sha256()

    def record(rel: str) -> None:
        full = src_dir / rel
        st = os.lstat(full)
        mode = st.st_mode
        if stat.S_ISLNK(mode):
            h.update(f"l {rel}\0{os.readlink(full)}\0".encode())
        elif stat.S_ISDIR(mode):
            h.update(f"d {rel}\0".encode())
            for name in sorted(os.listdir(full)):
                record(f"{rel}/{name}")
        else:
            x = "x" if mode & stat.S_IXUSR else "-"
            h.update(f"f {rel}\0{x}\0{st.st_size}\0".encode())
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)

    record(relpath.rstrip("/"))
    return h.hexdigest()


def archive_name(blob_id: str) -> str:
    return f"{blob_id}{BLOB_SUFFIX}"


def patch_name(sha256: str) -> str:
    return f"{sha256}{PATCH_SUFFIX}"


def object_name(ref: str) -> str:
    """A stored object's file name from a reference that may predate
    patches: bench state files written before version 3 pinned bare blob
    ids, which only ever named archives."""
    return ref if ref.endswith(OBJECT_SUFFIXES) else archive_name(ref)


@dataclass(frozen=True)
class StoredObject:
    """One file under ``blobs/``: what a fetch moves and verifies, what
    retention counts, and what the sweep keeps or deletes."""

    name: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Delta:
    """A blob stored as patches against another blob's archive.

    The base's sha256 and size are repeated here rather than looked up in
    the base's own entry: a remote consumer fetches and verifies the base
    from this entry alone, and the entry that first published the base may
    be pruned on the remote by then.
    """

    base_id: str
    base_sha256: str
    base_bytes: int
    plan: DeltaPlan

    def to_json(self) -> dict:
        return {
            "base": {
                "id": self.base_id,
                "sha256": self.base_sha256,
                "bytes": self.base_bytes,
            },
            **self.plan.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict, where: str) -> Delta:
        if not isinstance(data, dict) or not isinstance(data.get("base"), dict):
            raise BusError(f"bus entry {where} has a malformed delta: {data!r}")
        base = data["base"]
        if set(base) != {"id", "sha256", "bytes"} or not all(
            isinstance(base[k], t) for k, t in (("id", str), ("sha256", str))
        ):
            raise BusError(f"bus entry {where} has a malformed delta base: {base!r}")
        if not isinstance(base["bytes"], int) or isinstance(base["bytes"], bool):
            raise BusError(f"bus entry {where} has a malformed delta base: {base!r}")
        try:
            plan = DeltaPlan.from_json({k: v for k, v in data.items() if k != "base"})
        except DeltaError as e:
            raise BusError(f"bus entry {where} has a malformed delta: {e}") from e
        return cls(base["id"], base["sha256"], base["bytes"], plan)


@dataclass(frozen=True)
class Blob:
    """One run set entry of a built commit, and how it is stored.

    ``path`` is the run set entry the blob unpacks to and ``id`` the tree
    hash that identifies the result. A blob stored whole has ``sha256`` and
    ``bytes`` of its archive, ``blobs/<id>.tar.zst``. A blob stored as a
    delta has them empty and ``delta`` set: the archive is reconstructed
    from the base and the patches, and nothing named ``<id>.tar.zst`` need
    exist. A version 1 entry reads back as one whole blob with an empty path
    and the archive's sha256 for an id, since that is the only name it had.
    """

    path: str
    id: str
    sha256: str = ""
    bytes: int = 0
    delta: Delta | None = None

    @property
    def is_delta(self) -> bool:
        return self.delta is not None

    def objects(self) -> list[StoredObject]:
        """The files this blob needs, base first for a delta. Patches shared
        between blocks appear once."""
        if self.delta is None:
            return [StoredObject(archive_name(self.id), self.sha256, self.bytes)]
        d = self.delta
        return [
            StoredObject(archive_name(d.base_id), d.base_sha256, d.base_bytes),
            *(StoredObject(patch_name(s), s, n) for s, n in d.plan.patches().items()),
        ]

    def to_json(self) -> dict:
        d = {"path": self.path, "id": self.id}
        if self.delta is None:
            d["archive"] = {"sha256": self.sha256, "bytes": self.bytes}
        else:
            d["delta"] = self.delta.to_json()
        return d

    @classmethod
    def from_json(cls, item, where: str) -> Blob:
        if not isinstance(item, dict):
            raise BusError(f"bus entry {where} has a malformed blob: {item!r}")
        keys = set(item)
        if keys == {"path", "id", "sha256", "bytes"}:
            # Version 2: always an archive, fields flat.
            return cls(**item)
        if keys == {"path", "id", "archive"} and isinstance(item["archive"], dict):
            arc = item["archive"]
            if set(arc) == {"sha256", "bytes"}:
                return cls(item["path"], item["id"], arc["sha256"], arc["bytes"])
        if keys == {"path", "id", "delta"}:
            return cls(
                item["path"],
                item["id"],
                delta=Delta.from_json(item["delta"], where),
            )
        raise BusError(f"bus entry {where} has a malformed blob: {item!r}")


def _parse_blobs(raw, where: str) -> list[Blob]:
    if not isinstance(raw, list) or not raw:
        raise BusError(f"bus entry {where} names no blobs")
    return [Blob.from_json(item, where) for item in raw]


def distinct_objects(blobs) -> dict[str, StoredObject]:
    """name -> object over every blob given, each object once."""
    out: dict[str, StoredObject] = {}
    for blob in blobs:
        for obj in blob.objects():
            out.setdefault(obj.name, obj)
    return out


@dataclass
class Entry:
    """One built commit, ready to bench: a manifest over blobs.

    Key names match the store's columns, so a consumer can write the commits
    row straight from this. ``embedder_id`` defaults to 0 so an entry written
    before keys had two parts reads back as the scalar key it always was; the
    commit fields stay the engine's own commit, which is what names the entry.

    ``embedder`` is the outer commit a two-coordinate key was built under
    (``hash``, ``commit_id``, ``title`` of the chromium roll CL) and ``pins``
    what was pinned beneath it (``{"src/v8": hash}``); both are empty for an
    engine built from its own checkout, and absent from the entries such an
    engine wrote before the fields existed.

    ``blobs`` is in run set order, one per entry. Unpacking them all into one
    directory is the run root.
    """

    engine: str
    commit_id: int
    hash: str
    date: str
    timestamp: int
    title: str
    build_cfg_hash: str
    blobs: list[Blob]
    embedder_id: int = 0
    embedder: dict = field(default_factory=dict)
    pins: dict = field(default_factory=dict)
    builder: dict = field(default_factory=dict)
    built_at: int = 0
    build_secs: int = 0
    version: int = ENTRY_VERSION

    @property
    def key(self) -> CommitKey:
        return CommitKey(self.embedder_id, self.commit_id)

    @property
    def embedder_hash(self) -> str:
        return str(self.embedder.get("hash", "")) if self.embedder else ""

    @property
    def blob_bytes(self) -> int:
        """Bytes of the distinct stored objects this entry needs. A delta
        entry counts its base here: it is what a cold consumer fetches."""
        return sum(o.bytes for o in distinct_objects(self.blobs).values())

    def objects(self) -> list[StoredObject]:
        """Every stored object this entry needs, each once, in blob order."""
        return list(distinct_objects(self.blobs).values())

    def to_json(self) -> str:
        data = asdict(self)
        data["blobs"] = [b.to_json() for b in self.blobs]
        return json.dumps(data, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, where: str = "") -> Entry:
        try:
            data = json.loads(text)
        except ValueError as e:
            raise BusError(f"unreadable bus entry {where}: {e}") from e
        # Valid JSON that is not an object: null, a list, a bare number. Every
        # other malformed case here is a BusError, which a consumer survives,
        # and .get on one of these is an AttributeError, which it does not.
        if not isinstance(data, dict):
            raise BusError(f"bus entry {where} is not an object")
        version = data.get("version")
        if version not in READABLE_ENTRY_VERSIONS:
            raise BusError(
                f"bus entry {where} is version {version}, this slipstream reads "
                f"versions {list(READABLE_ENTRY_VERSIONS)}; upgrade the consumer"
            )
        data = dict(data)
        if version == 1:
            # One payload, named by its key and identified by its sha256. It
            # keeps version 1 so paths resolve to where that payload is until
            # the builder migrates it.
            if "blob_sha256" in data and "blob_bytes" in data:
                sha = data["blob_sha256"]
                data["blobs"] = [Blob("", sha, sha, data["blob_bytes"])]
        elif "blobs" in data:
            data["blobs"] = _parse_blobs(data["blobs"], where)
        known = {f for f in cls.__dataclass_fields__}
        # Every field the constructor requires, not just the interesting ones:
        # a missing one would otherwise raise TypeError, which a consumer does
        # not treat as a bad entry and so does not survive.
        required = {
            name
            for name, f in cls.__dataclass_fields__.items()
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required - data.keys()
        if missing:
            raise BusError(f"bus entry {where} is missing {sorted(missing)}")
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class BuilderState:
    """What the builder knows, published for consumers to read over ssh.

    ``updated_at`` is rewritten per entry rather than per cycle: a cycle
    drains, so it can span days, which would leave the file stalest exactly
    when the process is busiest. Without it a crashed builder leaves a
    plausible frontier and a null last_error forever.

    The key fields are written with ``CommitKey.to_json``: a bare int for a
    scalar key, so a v8 or jsc state file is byte-identical to one written
    before keys had two parts, and a pair otherwise.
    """

    KEY_FIELDS = ("frontier", "lowest_retained", "highest_dropped")

    frontier: CommitKey | None = None
    lowest_retained: CommitKey | None = None
    # The highest entry retention has ever deleted. A consumer whose cursor is
    # below it provably never benched that entry, which comparing against
    # lowest_retained does not establish: commit ids are not contiguous, so a
    # gap in them is not evidence of anything.
    highest_dropped: CommitKey | None = None
    # The most recent failures only, with the total beside them: consumers
    # read this file over ssh once per cycle, and an unbounded list grows for
    # the life of the engine.
    failed: list[dict] = field(default_factory=list)
    failed_total: int = 0
    publishing_paused_by_floor: bool = False
    last_error: str | None = None
    stalled_since: float | None = None
    # When a stalled engine may be attempted again. Persisted rather than held
    # in the process, because `build --once` from launchd is a supported
    # deployment and each invocation is a new process.
    stall_retry_after: float | None = None
    in_flight: dict | None = None
    updated_at: float = 0.0
    version: int = VERSION


@dataclass
class BenchState:
    """What a bencher knows. Separate from the builder's file because the two
    are separate processes and a shared file would clobber whichever wrote
    first.

    ``env`` records the inputs shared between the boxes -- slipstream version,
    OS, run configs, harness revisions -- with ``since``, the time those values
    took effect, so a divergence check reports when the two boxes parted rather
    than only that they currently agree.
    """

    KEY_FIELDS = ("cursor", "skipped_dropped")

    bot: str | None = None
    cursor: CommitKey | None = None
    lag: int = 0
    status_counts: dict = field(default_factory=dict)
    last_error: str | None = None
    stalled_since: float | None = None
    # When a stalled bencher may try again. Persisted for the same reason the
    # builder's is: `watch --once` from launchd is a new process each time.
    stall_retry_after: float | None = None
    # Highest entry this machine skipped because retention had already dropped
    # it. Its own record, because the builder's highest_dropped stops being
    # evidence the moment the cursor passes it.
    skipped_dropped: CommitKey | None = None
    # Carry the breaker across round-robin turns and daemon restarts.
    consecutive_failures: int = 0
    benching_paused_by_floor: bool = False
    in_flight: dict | None = None
    # The blob ids of the entry this machine last provisioned. On a box with
    # no topic of its own nothing else references a fetched blob, and the
    # sweep would reclaim it before the next entry could reuse it; this keeps
    # exactly one entry's worth per engine, which is the whole cache.
    blobs: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    updated_at: float = 0.0
    version: int = VERSION


def state_to_json(state) -> str:
    data = asdict(state)
    for name in type(state).KEY_FIELDS:
        if data[name] is not None:
            data[name] = CommitKey.of(data[name]).to_json()
    return json.dumps(data, indent=1, sort_keys=True)


class Bus:
    """A bus root on this machine."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()

    # --- paths ---

    def topic_dir(self, engine: str) -> Path:
        return self.root / "topics" / "builds" / engine

    def entry_path(self, engine: str, key) -> Path:
        return self.topic_dir(engine) / f"{CommitKey.of(key)}.json"

    @property
    def blobs_dir(self) -> Path:
        return self.root / "blobs"

    def blob_path(self, blob_id: str) -> Path:
        return self.blobs_dir / f"{blob_id}{BLOB_SUFFIX}"

    def has_blob(self, blob_id: str) -> bool:
        return self.blob_path(blob_id).exists()

    def legacy_blob_path(self, engine: str, key) -> Path:
        """Where a version 1 entry's payload is until migrated."""
        return (
            self.root
            / "blobs"
            / "builds"
            / engine
            / f"{CommitKey.of(key)}{BLOB_SUFFIX}"
        )

    def object_path(self, name: str) -> Path:
        """Where a stored object -- an archive or a patch -- is by file name."""
        return self.blobs_dir / name

    def has_object(self, name: str) -> bool:
        return self.object_path(name).exists()

    def entry_object_path(self, entry: Entry, obj: StoredObject) -> Path:
        """Where one of an entry's objects is in this root."""
        if entry.version == 1 and not self.has_object(obj.name):
            return self.legacy_blob_path(entry.engine, entry.key)
        return self.object_path(obj.name)

    def builder_state_path(self, engine: str) -> Path:
        return self.root / "state" / "builds" / f"{engine}.json"

    def bench_state_path(self, engine: str) -> Path:
        return self.root / "state" / "bench" / f"{engine}.json"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    def engines(self) -> list[str]:
        """Every engine with a topic in this root, configured or not.

        Blobs are shared across engines, so anything that counts references
        must see every topic that exists, not only the ones this process was
        told about.
        """
        try:
            return sorted(os.listdir(self.root / "topics" / "builds"))
        except FileNotFoundError:
            return []

    # --- entries ---

    def keys(self, engine: str) -> list[CommitKey]:
        """Published keys, ascending. Ignores tmp files still being written."""
        try:
            names = os.listdir(self.topic_dir(engine))
        except FileNotFoundError:
            return []
        keys = []
        for name in names:
            stem, dot, ext = name.partition(".")
            key = parse_key_stem(stem) if dot and ext == "json" else None
            if key is not None:
                keys.append(key)
        return sorted(keys)

    def keys_above(self, engine: str, cursor) -> list[CommitKey]:
        keys = self.keys(engine)
        if cursor is None:
            return keys
        cursor = CommitKey.of(cursor)
        return [k for k in keys if k > cursor]

    def read_entry(self, engine: str, key) -> Entry | None:
        """The entry, or None if retention dropped it between listing and now."""
        path = self.entry_path(engine, key)
        try:
            text = path.read_text()
        except FileNotFoundError:
            return None
        return Entry.from_json(text, str(path))

    def entries(self, engine: str) -> list[Entry]:
        """Every readable entry of the engine, ascending by key."""
        found = []
        for key in self.keys(engine):
            entry = self.read_entry(engine, key)
            if entry is not None:
                found.append(entry)
        return found

    def publish(self, entry: Entry) -> None:
        """Write the manifest. Its blobs must already be stored.

        Blobs first, always: the reverse lets a consumer read an entry whose
        blobs do not exist. A crash between the two leaves unreferenced blobs,
        which is the harmless direction -- the sweep reclaims them, and the
        builder's retry of that commit finds them already stored.
        """
        _atomic_write(self.entry_path(entry.engine, entry.key), entry.to_json())

    def tmp_object(self, name: str) -> Path:
        """Where a writer puts an object before ``store_object`` renames it
        into place.

        Under tmp/ on the same filesystem, uniquely named: a crash leaves it
        for gc, and two processes producing the same object cannot collide.
        """
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        return self.tmp_dir / f"{name}.tmp-{os.getpid()}-{time.time_ns():x}"

    def tmp_blob(self, blob_id: str) -> Path:
        return self.tmp_object(archive_name(blob_id))

    def store_object(self, tmp: Path, name: str) -> Path:
        """Rename a complete, verified object into the store."""
        dest = self.object_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dest)
        return dest

    def store_blob(self, tmp: Path, blob_id: str) -> Path:
        """Rename a complete, verified archive into the blob store."""
        return self.store_object(tmp, archive_name(blob_id))

    # --- state files ---

    def read_builder_state(self, engine: str) -> BuilderState:
        return _read_state(self.builder_state_path(engine), BuilderState)

    def write_builder_state(self, engine: str, state: BuilderState) -> None:
        state.updated_at = time.time()
        _atomic_write(self.builder_state_path(engine), state_to_json(state))

    def read_bench_state(self, engine: str) -> BenchState:
        return _read_state(self.bench_state_path(engine), BenchState)

    def write_bench_state(self, engine: str, state: BenchState) -> None:
        state.updated_at = time.time()
        _atomic_write(self.bench_state_path(engine), state_to_json(state))

    # --- retention ---

    def prune(self, engine: str, retain_bytes: float, keep=None) -> list[CommitKey]:
        """Drop the oldest entries until the engine's blobs fit the budget.

        A byte budget rather than a count: artifact size per commit is what
        matters. The footprint is the distinct blobs the retained entries
        name, so an entry whose blobs are all shared with a newer one costs
        nothing to keep and dropping it frees nothing. Only manifests are
        deleted here; ``sweep_blobs`` then reclaims whatever no manifest in
        the root names, across every engine, since blobs are shared between
        them. Deleting a manifest before its blobs is the safe order: a crash
        between the two leaves blobs with no entry, which the next sweep
        reclaims, where the reverse would leave an entry pointing at nothing.

        What is retained is always a contiguous run ending at the newest entry.
        Keeping a smaller older entry below a dropped one would fit more in the
        budget, but it would leave a hole that nothing can detect: a consumer
        finds out about dropped entries by comparing its cursor against
        ``lowest_retained``, which a surviving older entry holds down, so it
        would step over the hole in silence.

        ``keep`` lowers the floor of that run rather than exempting one entry,
        for the commit just published: a ``build --retry`` republishes below
        the frontier, so its entry is the oldest and would otherwise be deleted
        by the very cycle that produced it. Exempting it alone would leave
        exactly the hole described above, so everything from it upward is
        retained and the budget is overshot for one cycle. The next publish
        prunes normally, dropping it along with its neighbours and reporting it.

        Retention does not coordinate with consumers. Detection does that
        instead: lowest_retained above a consumer's cursor means entries were
        dropped unread, which bus status reports.
        """
        keep = CommitKey.of(keep) if keep is not None else None
        dropped = []
        retained: set[str] = set()
        used = 0.0
        over_budget = False
        for entry in reversed(self.entries(engine)):
            objects = distinct_objects(entry.blobs)
            size = sum(o.bytes for n, o in objects.items() if n not in retained)
            if not over_budget and used and used + size > retain_bytes:
                over_budget = True
            if not over_budget or (keep is not None and entry.key >= keep):
                retained.update(objects)
                used += size
                continue
            self.entry_path(engine, entry.key).unlink(missing_ok=True)
            dropped.append(entry.key)
        return sorted(dropped)

    def lowest_retained(self, engine: str) -> CommitKey | None:
        keys = self.keys(engine)
        return keys[0] if keys else None

    def footprint(self, engine: str) -> int:
        """Bytes of the distinct stored objects the engine's entries name.
        A base shared by a group of delta entries is counted once."""
        seen: dict[str, int] = {}
        for entry in self.entries(engine):
            for obj in entry.objects():
                seen.setdefault(obj.name, obj.bytes)
        return sum(seen.values())

    def referenced_objects(self) -> set[str]:
        """Every stored object's name something in this root still needs.

        Every manifest of every topic, plus what each bencher's state file
        says it holds: on a box with no topic that list is the only reference
        a fetched object has. A delta entry names its base, so the base
        outlives its own entry for as long as any delta against it is kept.
        """
        refs: set[str] = set()
        for engine in self.engines():
            for entry in self.entries(engine):
                refs.update(o.name for o in entry.objects())
        bench_dir = self.root / "state" / "bench"
        for path in sorted(bench_dir.glob("*.json")) if bench_dir.exists() else []:
            refs.update(object_name(r) for r in self.read_bench_state(path.stem).blobs)
        return refs

    def sweep_blobs(self) -> list[Path]:
        """Delete every stored object nothing references. Returns what went.

        The caller holds the machine lock: an object between being stored and
        its manifest being written, or between being fetched and unpacked,
        is unreferenced too, and only the lock says nobody is in that window.
        """
        refs = self.referenced_objects()
        removed = []
        for path in sorted(self.blobs_dir.iterdir()) if self.blobs_dir.exists() else []:
            if (
                path.is_file()
                and path.name.endswith(OBJECT_SUFFIXES)
                and path.name not in refs
            ):
                path.unlink(missing_ok=True)
                removed.append(path)
        return removed

    def gc(self) -> list[Path]:
        """Delete what nothing names: blobs left by a crash mid-publish or
        mid-prune, half-fetched archives, and abandoned unpack directories.
        Returns what it removed. The caller holds the machine lock, so nothing
        here can be in use.
        """
        removed = []
        # _atomic_write leaves these anywhere it is used, including under
        # state/, which nothing else looked at.
        for state_dir in ("builds", "bench"):
            d = self.root / "state" / state_dir
            for partial in sorted(d.glob("*.tmp-*")) if d.exists() else []:
                partial.unlink(missing_ok=True)
                removed.append(partial)
        # A kill mid-unpack leaves a full-size directory that no commit id
        # names. The consumer sweeps these too, but only on a cycle that gets
        # past its free-space floor -- which is the cycle this is standing in
        # for.
        roots = self.root / "roots"
        for partial in sorted(roots.glob("*/*.unpacking")) if roots.exists() else []:
            shutil.rmtree(partial, ignore_errors=True)
            removed.append(partial)
        # An interrupted fetch or packaging leaves hundreds of MB here.
        for partial in sorted(self.tmp_dir.glob("*")) if self.tmp_dir.exists() else []:
            if partial.is_file():
                partial.unlink(missing_ok=True)
                removed.append(partial)
        for engine in self.engines():
            # A crash between writing an entry's tmp file and renaming it leaks
            # one per crash into the topic, which keys() ignores and nothing
            # else looked at.
            for partial in sorted(self.topic_dir(engine).glob("*.tmp-*")):
                partial.unlink(missing_ok=True)
                removed.append(partial)
        removed.extend(self.sweep_blobs())
        # Version 1 payloads whose entry retention dropped before the builder
        # migrated them, and the per-engine directories once they are empty.
        legacy = self.root / "blobs" / "builds"
        for engine_dir in sorted(legacy.iterdir()) if legacy.exists() else []:
            published = set(self.keys(engine_dir.name))
            for path in sorted(engine_dir.iterdir()):
                key = (
                    parse_key_stem(path.name[: -len(BLOB_SUFFIX)])
                    if path.name.endswith(BLOB_SUFFIX)
                    else None
                )
                if key is None or key not in published:
                    path.unlink(missing_ok=True)
                    removed.append(path)
            _rmdir_if_empty(engine_dir)
        _rmdir_if_empty(legacy)
        return removed

    # --- migration ---

    def migrate(self) -> list[CommitKey]:
        """Rewrite version 1 entries as manifests over the blob store.

        The payload is hardlinked under its sha256 -- its id, since the
        archive is the only thing there is to name it by -- the entry is
        rewritten naming it, and the old path is removed. Each step is
        idempotent and a crash anywhere leaves a readable root: an entry still
        at version 1 resolves to the old path while it exists and to the store
        once it does not. A payload already missing is left alone; the entry
        reports it exactly as it did before. The caller holds the machine
        lock. Returns the keys rewritten.
        """
        migrated = []
        legacy_root = self.root / "blobs" / "builds"
        for engine in self.engines():
            for entry in self.entries(engine):
                if entry.version != 1:
                    continue
                (blob,) = entry.blobs
                legacy = self.legacy_blob_path(engine, entry.key)
                dest = self.blob_path(blob.id)
                if not dest.exists():
                    if not legacy.exists():
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(legacy, dest)
                    except OSError:
                        tmp = self.tmp_blob(blob.id)
                        shutil.copyfile(legacy, tmp)
                        os.replace(tmp, dest)
                entry.version = ENTRY_VERSION
                self.publish(entry)
                legacy.unlink(missing_ok=True)
                migrated.append(entry.key)
            _rmdir_if_empty(legacy_root / engine)
        _rmdir_if_empty(legacy_root)
        return migrated


def _rmdir_if_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def state_from_json(text: str, cls, where: str = ""):
    """Parse a state file's contents. Also used for a copy read over ssh."""
    try:
        data = json.loads(text)
    except ValueError as e:
        raise BusError(f"unreadable bus state {where}: {e}") from e
    if not isinstance(data, dict):
        raise BusError(f"bus state {where} is not an object")
    if data.get("version") != VERSION:
        raise BusError(
            f"bus state {where} is version {data.get('version')}, this "
            f"slipstream reads version {VERSION}"
        )
    known = {f for f in cls.__dataclass_fields__}
    state = cls(**{k: v for k, v in data.items() if k in known})
    for name in cls.KEY_FIELDS:
        try:
            setattr(state, name, CommitKey.from_json(getattr(state, name)))
        except (TypeError, ValueError) as e:
            raise BusError(f"bus state {where} has a bad {name}: {e}") from e
    return state


def _read_state(path: Path, cls):
    try:
        text = path.read_text()
    except FileNotFoundError:
        return cls()
    return state_from_json(text, cls, str(path))


# --- consumer cursors ---


def cursor_path(out_dir: Path, source_name: str, engine: str) -> Path:
    return out_dir / "cursors" / source_name / "builds" / engine


def read_cursor(path: Path) -> CommitKey | None:
    """The last key benched from this source, or None if never.

    A torn cursor is reported rather than guessed at: the caller falls back to
    the highest done commit for the engine, which is derivable and cheap.
    """
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    try:
        return CommitKey.parse(text)
    except ValueError as e:
        raise BusError(f"unreadable cursor {path}: {text!r}") from e


def write_cursor(path: Path, key) -> None:
    _atomic_write(path, f"{CommitKey.of(key)}\n")
