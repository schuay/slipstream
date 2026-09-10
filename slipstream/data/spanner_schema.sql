-- Copyright 2026 The slipstream developers
-- SPDX-License-Identifier: MIT

-- Spanner schema (GoogleSQL). This is the authoritative DDL for the tables
-- slipstream pushes into. The database feeds other perf frontends, so the
-- schema, including the stored trace_id, must not change.

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

CREATE INDEX IF NOT EXISTS slipstream_refresh_idx
    ON slipstream (bot, benchmark, commit_number);

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
    trace_id      INT64 NOT NULL AS (FARM_FINGERPRINT(CONCAT(
                      bot, '\x1f', benchmark, '\x1f', test, '\x1f',
                      submetric, '\x1f', variant))) STORED,
) PRIMARY KEY (bot, benchmark, test, submetric, variant, commit_number);

CREATE INDEX IF NOT EXISTS benchmarks_filter_idx
    ON benchmarks (bot, benchmark, commit_time);

CREATE INDEX IF NOT EXISTS benchmarks_trace_idx
    ON benchmarks (trace_id, commit_number);

CREATE TABLE IF NOT EXISTS meta (
    key   STRING(MAX) NOT NULL,
    value STRING(MAX)
) PRIMARY KEY (key);
