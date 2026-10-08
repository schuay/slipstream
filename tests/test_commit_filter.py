# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess

import pytest

from slipstream.collector import BenchCollector
from slipstream.commit_filter import V8CommitFilter
from slipstream.config import EngineConfig
from slipstream.models import CommitKey
from test_resolve import ID_REGEX, Repo, _resolver


@pytest.fixture
def policy():
    return V8CommitFilter('target_cpu = "arm64"\ntarget_os = "mac"', "arm64")


@pytest.mark.parametrize(
    "path",
    [
        "test/torque/test-torque.tq",
        "tools/metagen/metagen.py",
        "agents/rules/git-cl.md",
        "docs/readme.md",
        "src/compiler/OWNERS",
        "AUTHORS",
        "src/compiler/backend/ppc/code-generator-ppc.cc",
        "src/base/cpu/cpu-riscv.cc",
    ],
)
def test_ignored_paths(policy, path):
    assert policy.ignores_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/compiler/backend/arm64/code-generator-arm64.cc",
        "src/wasm/jump-table-assembler.cc",
        "BUILD.gn",
        "DEPS",
        "src/execution/arm64/simulator-logic-arm64.cc",
    ],
)
def test_runtime_and_simulator_changes_are_retained(policy, path):
    assert not policy.ignores_path(path)


def test_cross_build_keeps_host_and_target_backends():
    policy = V8CommitFilter('target_cpu = "arm64"', "x86_64")
    assert not policy.ignores_path("src/codegen/x64/assembler-x64.cc")
    assert not policy.ignores_path("src/codegen/arm64/assembler-arm64.cc")
    assert policy.ignores_path("src/codegen/ppc/assembler-ppc.cc")


def test_computed_cpu_disables_backend_filter():
    policy = V8CommitFilter("target_cpu = custom_cpu", "arm64")
    assert not policy.ignores_path("src/codegen/ppc/assembler-ppc.cc")


DEPS = """
vars = {
    'test_rev': 'old',
    'android_rev': 'old',
    'runtime_rev': 'old',
    'alias': Var('test_rev'),
}
deps = {
    'third_party/fuzztest/src': 'url@' + Var('alias'),
    'third_party/android_sdk/public': {'version': Var('android_rev')},
    'third_party/icu': 'url@' + Var('runtime_rev'),
}
hooks = []
"""


def test_test_and_android_dependency_vars_follow_references(policy):
    after = DEPS.replace("'test_rev': 'old'", "'test_rev': 'new'").replace(
        "'android_rev': 'old'", "'android_rev': 'new'"
    )
    assert policy.ignores_deps_change(DEPS, after)
    assert policy.ignores_deps_change(
        DEPS, DEPS.replace("'third_party/fuzztest/src':", "'test/new-suite':")
    )


@pytest.mark.parametrize(
    "after",
    [
        DEPS.replace("'runtime_rev': 'old'", "'runtime_rev': 'new'"),
        DEPS.replace("hooks = []", "hooks = [{'action': ['build.py']}]"),
        DEPS + "gclient_gn_args = ['test_rev']\n",
        DEPS + "unknown_setting = 'new'\n",
        "invalid python!",
        DEPS + "deps = {}\n",
    ],
)
def test_runtime_hooks_exports_and_unknown_deps_changes_are_kept(policy, after):
    assert not policy.ignores_deps_change(DEPS, after)


def test_variable_also_used_by_runtime_or_hooks_is_kept(policy):
    for suffix in (
        "hooks = [{'action': [Var('alias')]}]\n",
        "gclient_gn_args = ['test_rev']\n",
        "other = Var('test_rev')\n",
    ):
        before = DEPS + suffix
        assert not policy.ignores_deps_change(
            before, before.replace("'test_rev': 'old'", "'test_rev': 'new'")
        )


def test_android_target_keeps_android_rolls():
    policy = V8CommitFilter('target_cpu = "arm64"\ntarget_os = "android"', "arm64")
    assert not policy.ignores_deps_change(
        DEPS, DEPS.replace("'android_rev': 'old'", "'android_rev': 'new'")
    )


def test_hyphenated_deps_variable_names(policy):
    before = "vars = {'android_sdk_build-tools_version': 'old'}\ndeps = {'third_party/android_sdk/public': Var('android_sdk_build-tools_version')}"
    assert policy.ignores_deps_change(before, before.replace("'old'", "'new'"))


def test_unknown_calls_in_ignored_dependencies_are_kept(policy):
    assert not policy.ignores_deps_change(
        DEPS, DEPS.replace("Var('alias')", "update_build_flags()")
    )
    assert not policy.ignores_deps_change(
        DEPS, DEPS.replace("'test_rev': 'old'", "'test_rev': update_build_flags()")
    )


def test_parsing_never_executes_deps(policy, tmp_path):
    marker = tmp_path / "executed"
    before = DEPS + f"open({str(marker)!r}, 'w').write('oops')\n"
    assert policy.ignores_deps_change(
        before, before.replace("'test_rev': 'old'", "'test_rev': 'new'")
    )
    assert not marker.exists()


def test_real_history_skips_only_wholly_irrelevant_commits(config, tmp_path):
    repo = Repo(tmp_path / "v8")
    base = repo.commit(1000, "base", {"src/main.cc": "a", "DEPS": DEPS})
    tool = repo.commit(1001, "tool", {"tools/helper.py": "a"})
    repo.commit(
        1002,
        "test dependency roll",
        {"DEPS": DEPS.replace("'test_rev': 'old'", "'test_rev': 'new'")},
    )
    repo.commit(1003, "other CPU", {"src/codegen/ppc/assembler-ppc.cc": "a"})
    runtime = repo.commit(1004, "mixed", {"tools/helper.py": "b", "src/main.cc": "b"})
    # A rename from a measured path into tools must still be measured.
    repo.git("mv", "src/main.cc", "tools/main.cc")
    repo.git("commit", "-qm", "rename\n\nCr-Commit-Position: refs/heads/main@{#1005}")
    repo.git("fetch", "-q", "origin", "main")
    rename = repo.git("rev-parse", "HEAD")
    engine = config.engines["v8"] = EngineConfig(
        name="v8",
        src_dir=repo.path,
        build_cmd="true",
        binary_path="d8",
        id_regex=ID_REGEX,
        gn_args='target_cpu = "arm64"',
    )
    collector = BenchCollector(config, backup=False)
    assert collector.next_commit_after(engine, 1000)["hash"] == runtime
    assert [c["hash"] for c in collector.get_commit_list(engine, 1000, 1005)] == [
        base,
        runtime,
        rename,
    ]
    # An explicit starting point remains a requested baseline.
    assert collector.get_commit_list(engine, 1001, 1004)[0]["hash"] == tool


def test_failed_git_read_is_measured(config, tmp_path, monkeypatch):
    engine = EngineConfig(
        name="v8",
        src_dir=tmp_path,
        build_cmd="true",
        binary_path="d8",
        id_regex=ID_REGEX,
    )
    collector = BenchCollector(config, backup=False)
    monkeypatch.setattr(
        collector,
        "_run",
        lambda *a, **k: subprocess.CompletedProcess("git", 1, "", "error"),
    )
    assert collector.commit_is_relevant(engine, "abc")


def test_chromium_roll_baseline_survives_filter(config, tmp_path):
    v8, chrome = Repo(tmp_path / "v8"), Repo(tmp_path / "chrome")
    base = v8.commit(1000, "base", {"test/base.js": "a"})
    v8.commit(1001, "test only", {"test/test.js": "a"})
    head = v8.commit(1002, "runtime", {"src/runtime.cc": "a"})
    chrome.commit(5000, "base", {"DEPS": f"'v8_revision': '{base}'"})
    chrome.commit(5001, "roll", {"DEPS": f"'v8_revision': '{head}'"})
    resolver = _resolver(config, v8.path, chrome.path)
    resolver.inner.path_filter = ""
    first = resolver.next_after(CommitKey(5001, 0))
    assert first.key == CommitKey(5001, 1000)
    assert resolver.next_after(first.key).key == CommitKey(5001, 1002)
    assert resolver.for_key(CommitKey(5001, 1001)) is None
