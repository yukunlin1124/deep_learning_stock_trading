"""tqdm progress bars + a stdout/stderr Tee that keeps them out of log files.

Usage:
    from src.utils.progress import pbar, log, Tee

    for code in pbar(codes, desc="loading bars", unit="stock"):
        ...
        log(f"[skip] {code}")      # prints above the bar instead of mangling it

`Tee` mirrors everything to the terminal, but drops tqdm's in-place redraws
(carriage-return / cursor-move writes) from the log file so logs stay
readable: they keep the `log(...)` / `print(...)` lines and nothing else.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def pbar(iterable: Iterable[T] | None = None, **kwargs: Any) -> tqdm:
    """tqdm with project defaults (dynamic width, writes to current stderr)."""
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(iterable, **kwargs)


def n_walk_steps(udates: Any, first_idx: int, r: int, end: Any) -> int:
    """Number of r-day walk-forward steps an IL loop will take (for a bar total).

    Mirrors the `while i + r <= len(udates): if udates[i] > end: break` loops.
    """
    return sum(1 for i in range(first_idx, len(udates) - r + 1, r) if udates[i] <= end)


def log(msg: str = "") -> None:
    """print() that cooperates with any live tqdm bars."""
    tqdm.write(str(msg), file=sys.stdout)


class Tee:
    """Write a stream to BOTH the terminal and a log file.

    Progress-bar redraws (any write containing '\\r' or an ANSI cursor-move)
    go to the terminal only. Log output is line-buffered so that the bare
    newlines tqdm emits around bar closes / nested-bar moves can be told
    apart from real lines: an empty line that follows bar activity is
    skipped; every non-empty line is kept.
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self._buf = ""
        self._bar_dirty = False  # bar activity since the last logged line

    def write(self, data):
        self.stream.write(data)
        if "\r" in data or "\x1b[" in data:
            self._bar_dirty = True
            return len(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line == "" and self._bar_dirty:
                continue
            self.log_file.write(line + "\n")
            self._bar_dirty = False
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    # tqdm / other libs probe these to pick unicode vs ascii and colour support.
    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def fileno(self):
        return self.stream.fileno()
