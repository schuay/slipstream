# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Human process output, with literal text and explicit presentation."""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.text import Text

from .durability import durable_mkdir


class Message(str):
    """A string callback can carry severity and Rich spans without markup."""

    def __new__(cls, text: Text, level=logging.INFO):
        value = super().__new__(cls, text.plain)
        value.text = text
        value.level = level
        return value


def message(*parts, level=logging.INFO):
    return Message(Text.assemble(*parts), level)


def prefixed(prefix, text):
    styled = text.text if isinstance(text, Message) else Text(str(text))
    return Message(Text(prefix) + styled, getattr(text, "level", logging.INFO))


def warning(text):
    return message(text, level=logging.WARNING)


def error(text):
    return message(text, level=logging.ERROR)


def success(text):
    return message((text, "green"))


# Remove terminal controls even from subprocess diagnostics. Keep newlines so
# each physical line receives its own prefix in persistent output.
_CONTROLS = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x08\x0b-\x1f\x7f-\x9f]"
)


def _clean(text):
    return _CONTROLS.sub("", text)


class _Terminal:
    def __init__(self):
        self.console = Console(stderr=True, highlight=False, markup=False)
        self.lock = threading.RLock()
        self.partial = None

    def break_line(self):
        if self.partial is not None:
            self.console.print()
            self.partial = None


_terminal = _Terminal()


class _Formatter(logging.Formatter):
    def __init__(self, role):
        super().__init__()
        self.role = role

    def prefix(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        severity = f"{record.levelname}: " if record.levelno >= logging.WARNING else ""
        return f"{timestamp} {self.role} {severity}"

    def format(self, record):
        prefix = self.prefix(record)
        return "\n".join(
            prefix + line for line in _clean(record.getMessage()).split("\n")
        )


class _StreamHandler(logging.Handler):
    def emit(self, record):
        terminal = _terminal
        with terminal.lock:
            terminal.break_line()
            rendered = self.format(record)
            if not terminal.console.is_terminal or "NO_COLOR" in os.environ:
                terminal.console.file.write(rendered + "\n")
                terminal.console.file.flush()
                return
            original = record.msg
            text = (
                original.text if isinstance(original, Message) else Text(str(original))
            )
            # Sanitizing may change offsets; diagnostics with controls remain plain.
            if _clean(text.plain) != text.plain:
                text = Text(_clean(text.plain))
            lines = text.split("\n", allow_blank=True)
            prefix = self.formatter.prefix(record)
            color = "red" if record.levelno >= logging.ERROR else "yellow"
            for line in lines:
                heading = Text(prefix)
                boundary = len(prefix)
                if record.levelno >= logging.WARNING:
                    boundary -= len(record.levelname) + 2
                    heading.stylize(color, boundary)
                heading.stylize("dim", 0, boundary)
                terminal.console.print(heading + line, soft_wrap=True)


class EventLog:
    def __init__(self, role, path=None, *, mirror_path=None):
        self.role = role
        self.mirror_path = mirror_path
        self.logger = logging.Logger(f"slipstream.{role}", level=logging.INFO)
        if path is None:
            self.handler = _StreamHandler()
        else:
            durable_mkdir(path.parent)
            self.handler = RotatingFileHandler(
                path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        self.handler.setFormatter(_Formatter(role))
        self.logger.addHandler(self.handler)
        self.logger.propagate = False
        self._progress = None

    def __call__(self, text):
        level = getattr(text, "level", logging.INFO)
        if not self.logger.isEnabledFor(level):
            return
        record = self.logger.makeRecord(self.logger.name, level, "", 0, text, (), None)
        with _terminal.lock:
            self.logger.handle(record)
            self._mirror(record)

    def _mirror(self, record):
        if self.mirror_path is not None:
            with self.mirror_path.open("a", encoding="utf-8") as stream:
                stream.write(self.handler.format(record) + "\n")

    def error(self, text):
        self(error(text))

    def warning(self, text):
        self(warning(text))

    def close(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()

    @property
    def interactive(self):
        return (
            isinstance(self.handler, _StreamHandler) and _terminal.console.is_terminal
        )

    @contextmanager
    def progress(self, label):
        """Own one partial terminal line and one complete persistent outcome."""
        with _terminal.lock:
            self._progress = label
            if self.interactive:
                _terminal.break_line()
                _terminal.console.print(Text(label + " "), end="", soft_wrap=True)
                _terminal.partial = self
        try:
            yield self
        except BaseException as exc:
            self.finish(
                "INTERRUPTED"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "FAILED",
                failed=True,
            )
            raise
        finally:
            if self._progress is not None:
                self.finish("INTERRUPTED", failed=True)

    def dot(self):
        with _terminal.lock:
            if self.interactive and _terminal.partial is self:
                _terminal.console.print(".", end="")

    def finish(self, outcome, *, failed=False):
        with _terminal.lock:
            label, self._progress = self._progress, None
            if label is None:
                return
            text = message(
                label + " ",
                (outcome, "red" if failed else "green"),
                level=logging.ERROR if failed else logging.INFO,
            )
            if self.interactive and _terminal.partial is self:
                _terminal.partial = None
                suffix = Text(" " + outcome)
                if "NO_COLOR" not in os.environ:
                    suffix.stylize("red" if failed else "green")
                _terminal.console.print(suffix, soft_wrap=True)
                record = self.logger.makeRecord(
                    self.logger.name, text.level, "", 0, text, (), None
                )
                self._mirror(record)
            else:
                self(text)
