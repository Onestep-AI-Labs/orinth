"""Printing, and the rule that makes the CLI pipeable.

**stdout carries data; stderr carries everything else.** Progress lines, stage
announcements, warnings, and errors all go to stderr, so `orinth dataset ls
--json | jq` works while the user still sees what is happening. Getting this
wrong is not a cosmetic problem: a spinner byte in the middle of a JSON document
breaks the consumer, and the failure appears in the *other* program.

Written by hand rather than with `rich` for the same reason. `rich` wants to own
the terminal and measure its width; here stdout must stay pipe-clean while
stderr carries the animated part, and configuring a renderer not to render is
more work than the ~100 lines below.
"""

import json
import shutil
import sys
from collections.abc import Iterable, Sequence
from typing import Any

#: Set once from the parsed global flags, because threading three presentation
#: booleans through every command signature would be worse than a module-level
#: switch that only `main()` writes.
_QUIET = False
_COLOR = True
_HEADER = True


def configure(*, quiet: bool = False, color: bool = True, header: bool = True) -> None:
    global _QUIET, _COLOR, _HEADER
    _QUIET = quiet
    # Colour is off whenever stdout is not a terminal, and honours NO_COLOR.
    _COLOR = color and sys.stderr.isatty()
    _HEADER = header


def is_tty() -> bool:
    return sys.stderr.isatty()


def note(message: str) -> None:
    """Progress and commentary. Never stdout."""
    if _QUIET:
        return
    print(message, file=sys.stderr, flush=True)


def warn(message: str) -> None:
    print(_paint(f"! {message}", "33"), file=sys.stderr, flush=True)


def error(message: str, hint: str = "") -> None:
    print(_paint(f"error: {message}", "31"), file=sys.stderr, flush=True)
    if hint:
        print(f"  {hint}", file=sys.stderr, flush=True)


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def emit_json(payload: Any) -> None:
    """One JSON document on stdout, and nothing else."""
    _write(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def emit_jsonl(rows: Iterable[Any]) -> None:
    for row in rows:
        _write(json.dumps(row, ensure_ascii=False, default=str))


def _write(line: str) -> None:
    """Write one stdout line, treating a closed pipe as a normal ending.

    `orinth dataset ls | head -3` closes the pipe after three lines. Without
    this the CLI dies with a `BrokenPipeError` traceback, which looks like a
    crash in a command that did exactly what was asked.
    """
    try:
        print(line, flush=True)
    except BrokenPipeError:
        raise SystemExit(0) from None


def table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> None:
    """A padded column table, truncated to the terminal width.

    An empty table still prints its header. `orinth dataset ls` on an empty
    project should look like a table with no rows, not like a command that
    silently did nothing.
    """
    text_rows = [[_cell(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(cell))

    limit = shutil.get_terminal_size((200, 24)).columns if is_tty() else 10_000

    if _HEADER:
        _write(_truncate(_row([h.upper() for h in headers], widths), limit))
    for row in text_rows:
        _write(_truncate(_row(row, widths), limit))


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    parts = [
        cell.ljust(widths[index]) if index < len(cells) - 1 else cell
        for index, cell in enumerate(cells)
    ]
    return "  ".join(parts).rstrip()


def _truncate(line: str, limit: int) -> str:
    return line if len(line) <= limit else line[: max(0, limit - 1)] + "…"


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "-"
    return str(value)


def key_values(pairs: Sequence[tuple[str, Any]]) -> None:
    """The detail form: one `key: value` per line, keys padded to align."""
    width = max((len(key) for key, _ in pairs), default=0)
    for key, value in pairs:
        _write(f"{key.ljust(width)}  {_cell(value)}")
