# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""A browser for tests: load the page it is given, then do what the
environment says. ``FAKE_BROWSER`` is ``report`` (POST ``FAKE_REPORT`` to
/report on the page's origin, then stay open like a browser would),
``exit`` (die before reporting) or ``hang`` (never report)."""

import os
import sys
import time
import urllib.request

url = sys.argv[-1]
mode = os.environ.get("FAKE_BROWSER", "report")

urllib.request.urlopen(url).read()
if mode == "exit":
    sys.exit(3)
if mode == "report":
    origin = url.split("/index.html")[0]
    body = os.environ["FAKE_REPORT"].encode()
    urllib.request.urlopen(
        urllib.request.Request(origin + "/report", data=body, method="POST")
    )
time.sleep(60)
