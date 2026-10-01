-- Copyright 2026 The slipstream developers
-- SPDX-License-Identifier: MIT

-- Spanner schema (GoogleSQL). This is the authoritative DDL for the tables
-- slipstream pushes into; index changes are applied to the live database by
-- hand to match it. The database feeds other perf frontends, so the tables,
-- including the stored trace_id, must not change shape; indexes are
-- slipstream's to change.

CREATE TABLE IF NOT EXISTS slipstream (
    bot           STRING(MAX) NOT NULL,
    benchmark     STRING(MAX) NOT NULL,
    test          STRING(MAX) NOT NULL,
    metric        STRING(MAX) NOT NULL DEFAULT (''),
    variant       STRING(MAX) NOT NULL DEFAULT ('default'),
    platform      STRING(MAX) NOT NULL DEFAULT (''),
    commit_number INT64       NOT NULL,
    run           INT64       NOT NULL DEFAULT (1),
    commit_time   TIMESTAMP,
    git_hash      STRING(MAX),
    val           FLOAT64     NOT NULL,
    imported_at   TIMESTAMP   NOT NULL DEFAULT (CURRENT_TIMESTAMP())
) PRIMARY KEY (bot, benchmark, test, metric, variant, platform, commit_number, run);

-- What changed since the watermark: refresh reads only rows newer than it.
-- The key is monotonic, which hotspots writes in general; at a few thousand
-- rows per push that does not matter here.
CREATE INDEX IF NOT EXISTS slipstream_imported_at_idx
    ON slipstream (imported_at);

-- Aggregation seeks a (bot, benchmark) pair to a set of commits. The remaining
-- key columns come with every index entry; STORING covers the rest, so the
-- read never goes back to the table.
CREATE INDEX IF NOT EXISTS slipstream_group_idx
    ON slipstream (bot, benchmark, commit_number)
    STORING (commit_time, git_hash, val);

CREATE TABLE IF NOT EXISTS benchmarks (
    bot           STRING(MAX) NOT NULL,
    benchmark     STRING(MAX) NOT NULL,
    test          STRING(MAX) NOT NULL,
    submetric     STRING(MAX) NOT NULL DEFAULT (''),
    variant       STRING(MAX) NOT NULL,
    commit_number INT64       NOT NULL,
    commit_time   TIMESTAMP,
    git_hash      STRING(MAX),
    source        STRING(MAX) NOT NULL DEFAULT ('slipstream'),
    mean          FLOAT64,
    min           FLOAT64,
    max           FLOAT64,
    stdev         FLOAT64,
    count         INT64,
-- The separator is the four characters \x1f, not the control character:
-- that is how the live table was created, and the stored values (which the
-- frontends key on) cannot be recomputed without dropping the column.
    trace_id      INT64 NOT NULL AS (FARM_FINGERPRINT(CONCAT(
                      bot, '\\x1f', benchmark, '\\x1f', test, '\\x1f',
                      submetric, '\\x1f', variant))) STORED,
) PRIMARY KEY (bot, benchmark, test, submetric, variant, commit_number);

CREATE INDEX IF NOT EXISTS benchmarks_filter_idx
    ON benchmarks (bot, benchmark, commit_time);

CREATE INDEX IF NOT EXISTS benchmarks_trace_idx
    ON benchmarks (trace_id, commit_number);

CREATE TABLE IF NOT EXISTS meta (
    key   STRING(MAX) NOT NULL,
    value STRING(MAX)
) PRIMARY KEY (key);
