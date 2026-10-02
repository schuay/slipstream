# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from slipstream.collector import BenchCollector, FetchError
from slipstream.config import EngineConfig
from slipstream.models import CommitKey
from slipstream.resolve import BuildJob, EmbedderResolver, IdentityResolver


def _commit(cid):
    return {
        "hash": f"hash{cid}",
        "commit_id": cid,
        "date": "2026-09-06",
        "timestamp": 1757116800 + cid,
        "title": f"commit {cid}",
    }


class FakeCollector:
    """The three collector calls the identity resolver is built on."""

    def __init__(self, history):
        self.history = history
        self.fetched = []

    def head_commit_id(self, name, fetch=True):
        self.fetched.append(name)
        return max(self.history)

    def next_commit_after(self, engine, commit_id):
        later = [c for c in self.history if c > commit_id]
        return _commit(later[0]) if later else None

    def commit_metadata_for_id(self, engine, commit_id):
        return _commit(commit_id) if commit_id in self.history else None


@pytest.fixture
def resolver():
    engine = SimpleNamespace(name="v8")
    return IdentityResolver(FakeCollector([100, 101, 103]), engine)


class TestIdentityResolver:
    def test_the_next_job_is_the_next_commit_under_a_scalar_key(self, resolver):
        job = resolver.next_after(CommitKey(0, 100))
        assert job == BuildJob(
            key=CommitKey(0, 101), commit=_commit(101), checkout_hash="hash101"
        )
        assert job.pins == {} and job.embedder == {}
        assert job.hash == "hash101"

    def test_a_gap_in_history_is_skipped_not_invented(self, resolver):
        assert resolver.next_after(CommitKey(0, 101)).key == CommitKey(0, 103)

    def test_up_to_date_is_none(self, resolver):
        assert resolver.next_after(CommitKey(0, 103)) is None

    def test_for_key_names_exactly_that_commit_or_nothing(self, resolver):
        assert resolver.for_key(CommitKey(0, 101)).checkout_hash == "hash101"
        assert resolver.for_key(CommitKey(0, 102)) is None

    def test_fetch_goes_to_the_engines_checkout(self, resolver):
        resolver.fetch()
        assert resolver.collector.fetched == ["v8"]

    @pytest.mark.parametrize("method", ["next_after", "for_key"])
    def test_a_key_with_an_embedder_is_a_misconfiguration(self, resolver, method):
        """[build] from or --retry wrote a two-coordinate key for an engine
        whose series has one; say so rather than silently dropping the
        embedder and building the wrong thing."""
        with pytest.raises(ValueError, match="own checkout"):
            getattr(resolver, method)(CommitKey(7, 100))


class TestBuildJob:
    def test_is_immutable(self):
        job = BuildJob(key=CommitKey(0, 1), commit=_commit(1), checkout_hash="h")
        with pytest.raises(AttributeError):
            job.key = CommitKey(0, 2)  # type: ignore[misc]


# --- EmbedderResolver against two scratch git repos ---

ID_REGEX = r"^ *Cr-Commit-Position:.*#([0-9]+)"
ROLL_REGEX = r"'v8_revision': '([0-9a-f]{40})'"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@x",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@x",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    "GIT_CONFIG_NOSYSTEM": "1",
}


class Repo:
    """A git repo whose commits carry Cr-Commit-Position trailers, with
    ``origin`` pointing at itself so ``git fetch origin main`` works."""

    def __init__(self, path: Path, *, clone_of: Path | None = None):
        self.path = path
        self.env = {**os.environ, **_GIT_ENV, "HOME": str(path)}
        self.by_pos: dict[int, str] = {}
        if clone_of is None:
            path.mkdir(parents=True)
            self.git("init", "-q", "-b", "main")
            self.git("remote", "add", "origin", str(path))
        else:
            subprocess.run(
                ["git", "clone", "-q", str(clone_of), str(path)],
                env=self.env,
                check=True,
                capture_output=True,
            )

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.path,
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(self, pos: int, subject: str, files: dict[str, str]) -> str:
        for rel, text in files.items():
            p = self.path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            self.git("add", rel)
        self.git(
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            f"{subject}\n\nCr-Commit-Position: refs/heads/main@{{#{pos}}}",
        )
        sha = self.git("rev-parse", "HEAD")
        self.by_pos[pos] = sha
        self.git("fetch", "-q", "origin", "main")
        return sha


def _deps(v8_sha: str, extra: str = "") -> str:
    return f"deps = {{\n  'v8_revision': '{v8_sha}',\n{extra}}}\n"


@pytest.fixture
def repos(tmp_path):
    """V8 1001..1008 (1007 touches nothing under src/); chromium rolls it
    1001->1003, then 1003->1005, reverts, re-rolls 1003->1006, then
    1006->1008, with non-roll DEPS churn and unrelated commits between."""
    v8 = Repo(tmp_path / "v8")
    for pos in range(1001, 1009):
        rel = "README" if pos == 1007 else f"src/f{pos}.cc"
        v8.commit(pos, f"v8 change {pos}", {rel: f"{pos}\n"})
    V = v8.by_pos

    cr = Repo(tmp_path / "chromium")
    cr.commit(5001, "Initial", {"DEPS": _deps(V[1001]), "foo.txt": "a\n"})
    cr.commit(5002, "Unrelated", {"foo.txt": "b\n"})
    cr.commit(5003, "Roll V8 1001..1003", {"DEPS": _deps(V[1003])})
    cr.commit(5004, "Roll skia", {"DEPS": _deps(V[1003], "  'skia': 'x',\n")})
    cr.commit(5005, "Roll V8 1003..1005", {"DEPS": _deps(V[1005], "  'skia': 'x',\n")})
    cr.commit(5006, "Revert Roll V8", {"DEPS": _deps(V[1003], "  'skia': 'x',\n")})
    cr.commit(
        5007, "Reland Roll V8 1003..1006", {"DEPS": _deps(V[1006], "  'skia': 'x',\n")}
    )
    cr.commit(5008, "Roll V8 1006..1008", {"DEPS": _deps(V[1008], "  'skia': 'x',\n")})
    return v8, cr


def _resolver(config, v8_path: Path, cr_path: Path) -> EmbedderResolver:
    config.engines["v8"] = EngineConfig(
        name="v8",
        src_dir=v8_path,
        build_cmd="true",
        binary_path="d8",
        id_regex=ID_REGEX,
        path_filter="src",
        run_set=["d8"],
    )
    config.engines["chrome"] = EngineConfig(
        name="chrome",
        src_dir=cr_path,
        build_cmd="true",
        binary_path="chrome",
        id_regex=ID_REGEX,
        run_set=["chrome"],
    )
    return EmbedderResolver(
        BenchCollector(config, role="build"),
        config.engines["chrome"],
        config.engines["v8"],
        pin="src/v8",
        roll_file="DEPS",
        roll_regex=ROLL_REGEX,
    )


@pytest.fixture
def embedder(repos, config):
    v8, cr = repos
    return _resolver(config, v8.path, cr.path)


def _walk(resolver, start):
    keys, frontier = [], start
    while True:
        job = resolver.next_after(frontier)
        if job is None:
            return keys
        keys.append(job.key)
        frontier = job.key
        assert len(keys) < 50, "the walk did not terminate"


class TestEmbedderResolver:
    def test_the_whole_series(self, embedder):
        """Every roll expands to its inner range, the first point of each is
        the chromium-only step, the revert is skipped, the re-roll resumes
        at the frontier's V8 commit, and the inner path_filter thins the
        middle of a roll but never its base."""
        K = CommitKey
        assert _walk(embedder, K(5001, 0)) == [
            K(5003, 1001),
            K(5003, 1002),
            K(5003, 1003),
            K(5005, 1003),
            K(5005, 1004),
            K(5005, 1005),
            K(5007, 1005),
            K(5007, 1006),
            K(5008, 1006),
            K(5008, 1008),  # 1007 touches nothing under src/
        ]

    def test_a_job_says_how_to_build_it(self, embedder, repos):
        v8, cr = repos
        job = embedder.next_after(CommitKey(5001, 0))
        assert job.key == CommitKey(5003, 1001)
        assert job.checkout_hash == cr.by_pos[5003]
        assert job.pins == {"src/v8": v8.by_pos[1001]}
        assert job.commit["hash"] == v8.by_pos[1001]
        assert job.commit["title"] == "v8 change 1001"
        assert job.embedder == {
            "hash": cr.by_pos[5003],
            "commit_id": 5003,
            "title": "Roll V8 1001..1003",
        }

    def test_a_cold_start_on_a_roll_begins_at_its_old_pin(self, embedder):
        assert embedder.next_after(CommitKey(5005, 0)).key == CommitKey(5005, 1003)

    def test_a_cold_start_on_a_non_roll_scans_forward(self, embedder):
        assert embedder.next_after(CommitKey(5004, 0)).key == CommitKey(5005, 1003)

    def test_up_to_date(self, embedder):
        assert embedder.next_after(CommitKey(5008, 1008)) is None

    def test_for_key_names_a_point_on_the_series_or_nothing(self, embedder):
        assert embedder.for_key(CommitKey(5005, 1003)).key == CommitKey(5005, 1003)
        assert embedder.for_key(CommitKey(5008, 1008)).key == CommitKey(5008, 1008)
        assert embedder.for_key(CommitKey(5008, 1007)) is None  # filtered out
        assert embedder.for_key(CommitKey(5005, 1006)) is None  # not in that roll
        assert embedder.for_key(CommitKey(5006, 1003)) is None  # a revert
        assert embedder.for_key(CommitKey(5004, 1003)) is None  # not a roll

    def test_an_unknown_outer_position_is_an_error_not_up_to_date(self, embedder):
        with pytest.raises(ValueError, match="no commit at position 4242"):
            embedder.next_after(CommitKey(4242, 0))

    def test_a_roll_past_the_inner_checkout_waits_for_the_fetch(
        self, repos, config, tmp_path
    ):
        """Chromium can roll to a V8 commit the V8 checkout has not fetched
        yet. As far as the checkout can tell the roll is not forward, so it
        is skipped -- and since nothing was built the frontier stays put and
        the next cycle, after its fetch, finds the roll and expands it."""
        v8, cr = repos
        lagging = Repo(tmp_path / "v8_checkout", clone_of=v8.path)
        sha = v8.commit(1009, "v8 change 1009", {"src/f1009.cc": "1009\n"})
        cr.commit(5009, "Roll V8 1008..1009", {"DEPS": _deps(sha, "  'skia': 'x',\n")})
        resolver = _resolver(config, lagging.path, cr.path)
        assert resolver.next_after(CommitKey(5008, 1008)) is None
        resolver.fetch()
        assert _walk(resolver, CommitKey(5008, 1008)) == [
            CommitKey(5009, 1008),
            CommitKey(5009, 1009),
        ]

    def test_a_roll_to_a_commit_not_on_main_is_not_part_of_the_series(
        self, embedder, repos
    ):
        """A pin that is not a descendant of the previous one -- a revert, or
        a roll onto a branch -- is nothing V8's own series would walk; it is
        skipped, both as the frontier's roll and as a candidate."""
        v8, cr = repos
        cr.commit(5009, "Roll V8 to nowhere", {"DEPS": _deps("f" * 40)})
        assert embedder.next_after(CommitKey(5008, 1008)) is None
        assert embedder.next_after(CommitKey(5009, 0)) is None
        assert embedder.for_key(CommitKey(5009, 1008)) is None

    def test_fetch_touches_both_checkouts(self, embedder, repos):
        v8, cr = repos
        embedder.fetch()  # origin is each repo itself; a failure would raise
        cr.git("remote", "set-url", "origin", str(cr.path / "nowhere"))
        with pytest.raises(FetchError):
            embedder.fetch()
