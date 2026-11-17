# phlop/timing/severity.py
#
# Terminal color for "how bad is this", shared by the scope timer printers.
# Each printer decides the level against its own baseline, eg relative spread
# across runs, an event set hint, a share of the total; this only colors.
# PHLOP_TERM_COLORS=false, or stdout not a terminal, disables color.

from __future__ import annotations

import json
import os
import sys

TERM_COLORS = bool(json.loads(os.environ.get("PHLOP_TERM_COLORS", "true")))

OK, WARN, BAD = 0, 1, 2
_CODES = {WARN: "\033[93m", BAD: "\033[91m"}  # yellow, red
_RESET = "\033[0m"
_MARKS = {WARN: " (!)", BAD: " (!!)"}


def level(value, warn, bad):
    """OK, WARN or BAD for value against thresholds, higher is worse.
    None, eg not measured, is OK"""
    if value is None:
        return OK
    return BAD if value >= bad else WARN if value >= warn else OK


class Colors:
    def __init__(self, enabled=None, stream=None):
        if enabled is None:
            stream = stream or sys.stdout
            enabled = TERM_COLORS and stream.isatty()
        self.enabled = enabled

    def __call__(self, text, lvl):
        """text colored for lvl, reset after so nothing leaks past it.
        Without color, eg into a file or CI log, marked (!) or (!!) instead"""
        if lvl not in _CODES:
            return str(text)
        if not self.enabled:
            return f"{text}{_MARKS[lvl]}"
        return f"{_CODES[lvl]}{text}{_RESET}"

    def legend(self, warn, bad):
        """one line saying what the levels mean, eg legend(">=5%", ">=20%")"""
        if self.enabled:
            return f"colors: {self('yellow', WARN)} {warn}, {self('red', BAD)} {bad}"
        return f"marks: (!) {warn}, (!!) {bad}"
