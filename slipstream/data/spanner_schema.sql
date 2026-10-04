-- Copyright 2026 The slipstream developers
-- SPDX-License-Identifier: MIT

-- Spanner schema (GoogleSQL). This is the authoritative DDL for the tables
-- slipstream pushes into; index changes are applied to the live database by
-- hand to match it. The database feeds other perf frontends through
-- benchmarks and benchmarks_v2, so those tables, including their stored
-- trace_id, must not change shape; the staging side is slipstream's own.
--
-- Staging is in transition. samples, commits, dirty_groups and imports are
-- the current design: the key of samples is the aggregation group followed
-- by what varies within it, commit identity lives once in commits, and a
-- push leaves the groups it touched in dirty_groups for refresh to drain.
-- slipstream and meta are the previous design, still written alongside
-- while the new tables prove themselves; see the push target's "write" and
-- "aggregate_from" settings. The aggregate side is in the same kind of
-- transition from benchmarks to benchmarks_v2 ("aggregate_into").

-- One commit of one engine, possibly measured inside an embedder (chrome
-- around V8, Safari around JSC). Written by push in the same commit as its
-- samples; never deleted, since it is per engine rather than per bot. The
-- one place that links an inner commit to the outer one it ran inside.
CREATE TABLE IF NOT EXISTS commits (
    engine          STRING(MAX) NOT NULL,
    embedder_number INT64       NOT NULL,
    commit_number   INT64       NOT NULL,
    git_hash        STRING(MAX),
    commit_time     TIMESTAMP,
    title           STRING(MAX),
    embedder_hash   STRING(MAX),
    embedder_title  STRING(MAX)
) PRIMARY KEY (engine, embedder_number, commit_number);

-- One measured value. Refresh of a group and a bot's rebuild are prefix
-- range scans on the base table, so there is no secondary index. suite is
-- the short name ("js3", no alias), test is named as in benchmarks ("Air",
-- "Overall"), and variant is the run's own flags string ("default"); the
-- frontend labels are derived at refresh time, see frontend_benchmark and
-- frontend_variant in spanner.py. measured_at is the bench's own clock, NULL
-- when unknown; imported_at is a commit timestamp kept for audit, not for
-- change detection.
CREATE TABLE IF NOT EXISTS samples (
    bot             STRING(MAX) NOT NULL,
    suite           STRING(MAX) NOT NULL,
    engine          STRING(MAX) NOT NULL,
    variant         STRING(MAX) NOT NULL,
    embedder_number INT64       NOT NULL,
    commit_number   INT64       NOT NULL,
    test            STRING(MAX) NOT NULL,
    metric          STRING(MAX) NOT NULL,
    run             INT64       NOT NULL,
    value           FLOAT64     NOT NULL,
    measured_at     TIMESTAMP,
    imported_at     TIMESTAMP   NOT NULL OPTIONS (allow_commit_timestamp = true)
) PRIMARY KEY (bot, suite, engine, variant, embedder_number, commit_number, test, metric, run);

-- Groups that gained samples and have not been re-aggregated since. Written
-- in the same commit as the samples; refresh reads them, aggregates, and
-- deletes only rows no newer than what it read.
CREATE TABLE IF NOT EXISTS dirty_groups (
    bot             STRING(MAX) NOT NULL,
    suite           STRING(MAX) NOT NULL,
    engine          STRING(MAX) NOT NULL,
    variant         STRING(MAX) NOT NULL,
    embedder_number INT64       NOT NULL,
    commit_number   INT64       NOT NULL,
    dirtied_at      TIMESTAMP   NOT NULL OPTIONS (allow_commit_timestamp = true)
) PRIMARY KEY (bot, suite, engine, variant, embedder_number, commit_number);

-- One push attempt, from its first chunk to its last. A row without
-- finished_at is an interrupted import, and refresh waits for it to be
-- retried. Also the push ledger.
CREATE TABLE IF NOT EXISTS imports (
    attempt     STRING(MAX) NOT NULL,
    bot         STRING(MAX) NOT NULL,
    digest      STRING(MAX) NOT NULL,
    started_at  TIMESTAMP   NOT NULL OPTIONS (allow_commit_timestamp = true),
    finished_at TIMESTAMP   OPTIONS (allow_commit_timestamp = true),
    row_count   INT64
) PRIMARY KEY (attempt);

-- Previous staging design: group identity is encoded in the variant string
-- ("v8 (v8_default)") and the aliased benchmark name, change detection is a
-- watermark over imported_at in meta, and two indexes reorder the table for
-- aggregation. Kept and written to until the tables above have proven
-- themselves; then dropped together with its indexes and meta.
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

-- The aggregate table with an embedder coordinate: a browser build is a
-- (chromium position | STP build, V8 | WebKit commit) pair, and the same
-- inner commit is measured under two embedders at every roll boundary on
-- purpose, which benchmarks' key cannot hold. embedder_number is 0 for an
-- engine that is its own embedder. The trace_id expression is byte for
-- byte the one above, so every existing series keeps its id; a series has
-- no embedder coordinate, its points do. Written beside benchmarks until
-- the frontends read this table by name, then benchmarks is dropped; see
-- the push target's "aggregate_into" setting.
CREATE TABLE IF NOT EXISTS benchmarks_v2 (
    bot             STRING(MAX) NOT NULL,
    benchmark       STRING(MAX) NOT NULL,
    test            STRING(MAX) NOT NULL,
    submetric       STRING(MAX) NOT NULL DEFAULT (''),
    variant         STRING(MAX) NOT NULL,
    embedder_number INT64       NOT NULL,
    commit_number   INT64       NOT NULL,
    commit_time     TIMESTAMP,
    git_hash        STRING(MAX),
    embedder_hash   STRING(MAX),
    source          STRING(MAX) NOT NULL DEFAULT ('slipstream'),
    mean            FLOAT64,
    min             FLOAT64,
    max             FLOAT64,
    stdev           FLOAT64,
    count           INT64,
    trace_id        INT64 NOT NULL AS (FARM_FINGERPRINT(CONCAT(
                        bot, '\\x1f', benchmark, '\\x1f', test, '\\x1f',
                        submetric, '\\x1f', variant))) STORED,
) PRIMARY KEY (bot, benchmark, test, submetric, variant, embedder_number, commit_number);

CREATE INDEX IF NOT EXISTS benchmarks_v2_filter_idx
    ON benchmarks_v2 (bot, benchmark, commit_time);

CREATE INDEX IF NOT EXISTS benchmarks_v2_trace_idx
    ON benchmarks_v2 (trace_id, embedder_number, commit_number);

CREATE TABLE IF NOT EXISTS meta (
    key   STRING(MAX) NOT NULL,
    value STRING(MAX)
) PRIMARY KEY (key);
