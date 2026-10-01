# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Immutable delivery units. CSV remains the spool wire format."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass

COLUMNS = (
    "engine",
    "platform",
    "commit_id",
    "suite",
    "flags",
    "benchmark",
    "metric",
    "run",
    "score",
    "timestamp",
    "git_hash",
    "commit_date",
    "commit_timestamp",
    "commit_title",
)


def to_csv(rows) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    return stream.getvalue()


def parse_csv(text: str) -> tuple[tuple, ...]:
    reader = csv.reader(io.StringIO(text, newline=""))
    if tuple(next(reader, ())) != COLUMNS:
        raise ValueError("invalid spool CSV header")
    rows = []
    for row in reader:
        if len(row) != len(COLUMNS):
            raise ValueError("invalid spool CSV row width")
        if any(not row[i].strip() for i in (0, 1, 3, 4, 5, 10, 12)):
            raise ValueError("missing score identity or commit metadata")
        if int(row[2]) < 0 or int(row[7]) < 1 or not math.isfinite(float(row[8])):
            raise ValueError("invalid commit, run or score")
        int(row[9])
        int(row[12])
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True)
class Batch:
    source: str
    bot: str
    unit: int
    rows: tuple[tuple, ...]
    digest: str
    size: int
    raw_csv: str | None = None

    @classmethod
    def local(cls, source, bot, unit, rows):
        rows = tuple(tuple(r) for r in rows)
        wire = to_csv(rows)
        # Validate metadata before any target side effects, including spool writes.
        parse_csv(wire)
        return cls(
            source,
            bot,
            unit,
            rows,
            hashlib.sha256(wire.encode()).hexdigest(),
            len(wire.encode()),
        )

    @classmethod
    def remote(cls, source, bot, unit, text):
        return cls(
            source,
            bot,
            unit,
            parse_csv(text),
            hashlib.sha256(text.encode()).hexdigest(),
            len(text.encode()),
            text,
        )

    def csv(self):
        return self.raw_csv if self.raw_csv is not None else to_csv(self.rows)


@dataclass(frozen=True)
class Receipt:
    target: str
    attempt: str
    digest: str
