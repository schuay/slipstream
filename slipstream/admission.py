# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""One per-commit boundary shared by git, manual and bus collection."""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from . import bench_window


class StopReason(str, Enum):
    COMPLETE = "complete"
    DEFERRED = "window deferred"
    SHUTDOWN = "shutdown"
    FAILURE = "failure"


@dataclass(frozen=True)
class CollectionResult:
    benched: int = 0
    error: str | None = None
    reason: StopReason = StopReason.COMPLETE


class AdmissionStopped(Exception):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class Admission:
    def __init__(
        self, config, machine, *, wait=False, dry_run=False, log=lambda msg: None
    ):
        self.config = config
        self.machine = machine
        self.wait = wait
        self.dry_run = dry_run
        self.log = log

    def decision(self):
        return bench_window.current(self.config)

    def check(self, should_stop):
        if should_stop():
            raise AdmissionStopped(StopReason.SHUTDOWN, "shutdown requested")
        decision = self.decision()
        if not decision.allowed:
            raise AdmissionStopped(
                StopReason.DEFERRED, decision.error or decision.reason
            )

    @contextmanager
    def commit(self, job, should_stop=lambda: False):
        self.check(should_stop)
        if self.dry_run:
            yield
            return
        try:
            cancellation = None

            def cancelled():
                nonlocal cancellation
                try:
                    self.check(should_stop)
                except AdmissionStopped as exc:
                    cancellation = exc
                    return True
                return False

            if not self.machine.acquire(
                cancelled, wait=self.wait, log=self.log, job=job
            ):
                if cancellation is not None:
                    raise cancellation
                self.check(should_stop)
                raise AdmissionStopped(StopReason.SHUTDOWN, "lock wait cancelled")
            # Final race boundary; never checked during the commit.
            self.check(should_stop)
            yield
        finally:
            self.machine.release()
