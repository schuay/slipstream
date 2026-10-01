# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Plain standard logging with one sink and bounded optional file retention."""

from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler

from .durability import durable_mkdir


class EventLog:
    def __init__(self, role, path=None):
        self.logger = logging.Logger(f"slipstream.{role}", level=logging.INFO)
        if path is None:
            self.handler = logging.StreamHandler(sys.stderr)
        else:
            durable_mkdir(path.parent)
            self.handler = RotatingFileHandler(
                path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        self.logger.addHandler(self.handler)
        self.logger.propagate = False

    def __call__(self, message):
        # Source/target adapters retain a simple string event callback. This
        # shared CLI boundary supplies standard levels and presentation.
        level = (
            logging.WARNING
            if re.search(r"\b(failed|incomplete|blocked|not ready)\b", message)
            else logging.INFO
        )
        self.logger.log(level, message)

    def error(self, message):
        self.logger.error(message)

    def close(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
