# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""The interchange CSV shared by ``import``, ``export`` and ``analyze <csv>``.

This is the format skiz wrote before the SQLite store, and the only one that
crosses machines by hand. It grew a trailing ``bot`` column when one db came
to mean one bot; older files have six columns, and the oldest four. The
column is last so that header detection by first field still works.

Not to be confused with ``push._COLUMNS``, the export the push targets
consume: there the bot comes from the delivering session, so a column would be
dead weight and could disagree with the relay's configured name.
"""

from __future__ import annotations

FIELDS_7COL = [
    "b_type",
    "flags",
    "benchmark",
    "score_type",
    "commit_id",
    "score",
    "bot",
]
FIELDS_6COL = FIELDS_7COL[:-1]
FIELDS_4COL = ["benchmark", "score_type", "commit_id", "score"]

_BY_WIDTH = {7: FIELDS_7COL, 6: FIELDS_6COL, 4: FIELDS_4COL}


def sniff(first_row: list[str]) -> tuple[list[str] | None, bool]:
    """Read the first row of a compat CSV.

    Returns the field names to hand csv.DictReader (None when the row is a
    header) and whether the file carries a bot column. Raises ValueError on a
    width nothing has ever written.
    """
    fields = [f.strip() for f in first_row]
    if fields and fields[0] in ("b_type", "benchmark"):
        return None, "bot" in fields
    names = _BY_WIDTH.get(len(fields))
    if names is None:
        raise ValueError(f"unrecognized CSV format ({len(fields)} columns)")
    return list(names), len(fields) == 7
