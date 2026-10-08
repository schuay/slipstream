# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Conservative V8 commit selection for the benchmark build targets."""

from __future__ import annotations

import ast
import platform
import re
from dataclasses import dataclass


_ARCHES = {
    "arm",
    "arm64",
    "ia32",
    "x64",
    "ppc",
    "s390",
    "riscv",
    "riscv32",
    "riscv64",
    "loong64",
    "mips64",
}
_BACKENDS = {
    "builtins",
    "codegen",
    "compiler/backend",
    "deoptimizer",
    "diagnostics",
    "execution",
    "maglev",
    "regexp",
    "wasm/baseline",
}
_ALIASES = {"x86_64": "x64", "aarch64": "arm64", "ppc64": "ppc", "s390x": "s390"}


@dataclass(frozen=True)
class V8CommitFilter:
    gn_args: str
    host_cpu: str

    def _arg(self, name: str) -> str | None:
        text = "\n".join(line.split("#", 1)[0] for line in self.gn_args.splitlines())
        matches = re.findall(rf"\b{re.escape(name)}\s*=\s*([^\n]+)", text)
        return matches[-1].strip() if matches else None

    def _cpus(self) -> set[str] | None:
        host = _ALIASES.get(self.host_cpu, self.host_cpu)
        cpus = {host}
        for name in ("target_cpu", "v8_target_cpu", "v8_current_cpu"):
            value = self._arg(name)
            if value is not None:
                if not re.fullmatch(r'"[a-z0-9_]+"', value):
                    return None
                cpu = value.strip('"')
                cpus.add(_ALIASES.get(cpu, cpu))
        if not cpus <= _ARCHES:
            return None
        # RISC-V shares sources across its two targets.
        if cpus & {"riscv", "riscv32", "riscv64"}:
            cpus.update({"riscv", "riscv32", "riscv64"})
        return cpus

    def ignores_path(self, path: str) -> bool:
        if path.startswith(("tools/", "test/", "agents/", "docs/")):
            return True
        if path == "AUTHORS" or path.rsplit("/", 1)[-1] == "OWNERS":
            return True
        cpus = self._cpus()
        if cpus is None:
            return False
        for backend in _BACKENDS:
            prefix = f"src/{backend}/"
            if path.startswith(prefix):
                cpu = path[len(prefix) :].split("/", 1)[0]
                return cpu in _ARCHES and cpu not in cpus
        match = re.fullmatch(r"src/base/cpu/cpu-([a-z0-9]+)\.cc", path)
        return bool(match and match[1] in _ARCHES and match[1] not in cpus)

    def _ignores_dep(self, path: str) -> bool:
        if path.startswith(("tools/", "test/", "agents/")):
            return True
        for root in (
            "third_party/fuzztest",
            "third_party/googletest",
            "third_party/google_benchmark",
        ):
            if path == root or path.startswith(root + "/"):
                return True
        target_os = self._arg("target_os")
        non_android = target_os in ('"mac"', '"linux"', '"win"') or (
            target_os is None and platform.system() in ("Darwin", "Linux", "Windows")
        )
        return non_android and path.startswith("third_party/android_")

    def _deps_projection(self, text: str) -> str | None:
        """Remove ignored dependencies and vars used exclusively by them.

        Preserve everything else, including hooks and exported GN variables.
        AST comparison ignores formatting/comments without executing DEPS.
        """

        def ordinary_expression(root):
            for node in ast.walk(root):
                if isinstance(node, ast.Call):
                    if not (
                        isinstance(node.func, ast.Name)
                        and node.func.id in ("Var", "Str")
                        and len(node.args) == 1
                        and not node.keywords
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        return False
                if isinstance(
                    node,
                    (
                        ast.NamedExpr,
                        ast.Lambda,
                        ast.ListComp,
                        ast.DictComp,
                        ast.SetComp,
                        ast.GeneratorExp,
                    ),
                ):
                    return False
            return True

        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return None
        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in ("vars", "deps"):
                    if target.id in assignments or not isinstance(node.value, ast.Dict):
                        return None
                    keys = node.value.keys
                    if any(
                        not isinstance(k, ast.Constant) or not isinstance(k.value, str)
                        for k in keys
                    ):
                        return None
                    if len({k.value for k in keys}) != len(keys):
                        return None
                    assignments[target.id] = node.value
        if "deps" not in assignments:
            return None

        deps = assignments["deps"]
        removed = []
        retained_keys, retained_values = [], []
        for key, value in zip(deps.keys, deps.values):
            if self._ignores_dep(key.value):
                if not ordinary_expression(value):
                    return None
                removed.append(value)
            else:
                retained_keys.append(key)
                retained_values.append(value)
        deps.keys, deps.values = retained_keys, retained_values

        variables = assignments.get("vars")
        if variables is not None:
            values = {k.value: v for k, v in zip(variables.keys, variables.values)}

            def references(nodes):
                found = set()
                for root in nodes:
                    for node in ast.walk(root):
                        if isinstance(node, ast.Constant) and isinstance(
                            node.value, str
                        ):
                            found.add(node.value)
                            found.update(
                                re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", node.value)
                            )
                        elif isinstance(node, ast.Name):
                            found.add(node.id)
                return found & values.keys()

            candidates = references(removed)
            # Follow aliases and Var() references inside vars transitively.
            while True:
                expanded = candidates | references([values[k] for k in candidates])
                if expanded == candidates:
                    break
                candidates = expanded
            keys, vals = variables.keys, variables.values
            variables.keys, variables.values = [], []
            live = references([tree])
            # A variable definition outside the removable set is retained too.
            live.update(
                references([v for k, v in values.items() if k not in candidates])
            )
            while True:
                expanded = live | references([values[k] for k in live])
                if expanded == live:
                    break
                live = expanded
            removable = candidates - live
            if any(not ordinary_expression(values[k]) for k in removable):
                return None
            variables.keys = [k for k in keys if k.value not in removable]
            variables.values = [
                v for k, v in zip(keys, vals) if k.value not in removable
            ]
        return ast.dump(tree, include_attributes=False)

    def ignores_deps_change(self, before: str, after: str) -> bool:
        old = self._deps_projection(before)
        return old is not None and old == self._deps_projection(after)
