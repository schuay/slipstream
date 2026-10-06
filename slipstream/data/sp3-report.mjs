// Copyright 2026 The slipstream developers
// SPDX-License-Identifier: MIT
//
// Speedometer's results, posted to the bench server.
//
// Speedometer has no report protocol of its own: when the last iteration is
// done, MainBenchmarkClient.didFinishLastIteration(metrics) keeps the
// metrics on the page and nothing leaves it. The bench server appends this
// module to index.html after Speedometer's own scripts, so by the time it
// runs globalThis.benchmarkClient exists (release/3.x assigns it at module
// evaluation; main defers it to DOMContentLoaded, which is why there is a
// fallback). It wraps the two methods the runner calls at the end -- the
// same two crossbench and WebKit's run-benchmark hook -- and POSTs either
// {metrics} or {error} to /report on the page's own origin. The metrics
// serialise flat: Metric keeps parent, children and geomean non-enumerable.
//
// An error is posted before Speedometer handles it, so a broken run fails
// the config in seconds rather than at the suite timeout. The run is started
// by ?startAutomatically on the URL, not from here.

const REPORT_PATH = "/report";

function post(body) {
    return fetch(REPORT_PATH, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        keepalive: true,
    }).catch((e) => console.error("slipstream: report failed", e));
}

function describe(error) {
    return {
        message: String(error?.message ?? error),
        stack: typeof error?.stack === "string" ? error.stack : null,
    };
}

function install() {
    const client = globalThis.benchmarkClient;
    if (!client) {
        post({ error: { message: "slipstream: no benchmarkClient on the page", stack: null } });
        return;
    }
    const didFinishLastIteration = client.didFinishLastIteration;
    client.didFinishLastIteration = function (metrics, ...rest) {
        try {
            return didFinishLastIteration.call(this, metrics, ...rest);
        } finally {
            post({ metrics: metrics ?? this.metrics });
        }
    };
    const handleError = client.handleError;
    client.handleError = function (error, ...rest) {
        post({ error: describe(error) });
        return handleError.call(this, error, ...rest);
    };
}

if (globalThis.benchmarkClient || document.readyState !== "loading")
    install();
else
    document.addEventListener("DOMContentLoaded", install, { once: true });
