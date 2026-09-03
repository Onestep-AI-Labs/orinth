"""Shared pieces of a command module.

Imported by every group module and by none of them at dispatch time — `main.py`
imports one group, and that group imports this. Keeping it out of
`commands/__init__.py` is deliberate: an `__init__` that re-exported the groups
would import all of them, which is exactly what the two-phase dispatch avoids.
"""

import argparse
import time
from typing import Any

from app.cli import output
from app.cli.client import Client
from app.cli.config import CliConfig
from app.cli.errors import EXIT_UNREACHABLE, CliError


def make_client(config: CliConfig, globals_: argparse.Namespace) -> Client:
    client = Client(
        config.backend_url,
        source=config.backend.source,
        timeout=globals_.timeout or 60.0,
    )
    wait = getattr(globals_, "wait_backend", None)
    if wait:
        _wait_for(client, float(wait))
    return client


def _wait_for(client: Client, seconds: float) -> None:
    """Poll `/health` until it answers, so `orinth serve & orinth ...` works.

    Without this a CI script has to sleep an arbitrary amount and hope, which is
    either flaky or slow depending on which number was guessed.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if client.reachable():
            return
        time.sleep(0.5)
    raise CliError(
        f"Backend at {client.base_url} did not answer within {seconds:g}s",
        code=EXIT_UNREACHABLE,
        hint="Start one with `orinth serve`, or pass --backend URL.",
    )


def emit(rows: list[dict[str, Any]], columns: list[tuple[str, str]], globals_) -> None:
    """Render a list of records as JSON, JSONL, or a table.

    `columns` is `(header, key)` pairs, so the table's shape and the key order
    are declared once. `--json` always emits the *full* records rather than the
    projected columns: a table is a human summary, and silently dropping fields
    from machine output would make `--json` unusable for anything the table did
    not happen to show.
    """
    if globals_.json:
        output.emit_json(rows)
        return
    if globals_.jsonl:
        output.emit_jsonl(rows)
        return
    output.table(
        [[row.get(key) for _, key in columns] for row in rows],
        [header for header, _ in columns],
    )


def emit_one(record: dict[str, Any], pairs: list[tuple[str, Any]], globals_) -> None:
    if globals_.json or globals_.jsonl:
        output.emit_json(record)
        return
    output.key_values(pairs)


def add_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Re-declare the global flags on a subparser so they may appear anywhere.

    `orinth dataset ls --json` and `orinth --json dataset ls` must both work;
    users do not remember which side a flag lives on. These are parsed and
    discarded here — `main()` already read them off the full argv.
    """
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    parser.add_argument("--project", help=argparse.SUPPRESS)
    parser.add_argument("--workspace", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--wait-backend", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jsonl", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-header", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-q", "--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-v", "--verbose", action="store_true", help=argparse.SUPPRESS)
    return parser


def resolve_project(config: CliConfig, globals_: argparse.Namespace) -> str:
    return getattr(globals_, "project", None) or config.project.value
