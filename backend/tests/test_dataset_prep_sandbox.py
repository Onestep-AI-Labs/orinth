"""The sandbox the generated transform runs in.

The point of these is the *guards*, not the happy path. Every one of them is a
thing an LLM writes without being asked to — `import requests`, a loop that
never ends, `open(path, "w")` — and the whole case for running generated code
unattended rests on those failing loudly instead of quietly working.
"""

import pytest

from app.services.datasets.prep.sandbox import ALLOWED_IMPORTS, run_transform

ROWS = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def transform_source(body: str) -> str:
    return f"def transform(rows):\n{body}\n"


def test_a_transform_runs_and_returns_rows():
    result = run_transform(
        transform_source("    return [{'text': r['a'], 'label': r['b']} for r in rows]"), ROWS
    )
    assert result.ok
    assert result.rows == [{"text": "1", "label": "x"}, {"text": "2", "label": "y"}]


def test_the_allowed_stdlib_is_importable():
    # Every allowlisted module, imported at once. The C accelerators behind
    # `datetime` and `random` are why the driver pre-imports before gating:
    # a strict hook alone blocks `_datetime` and breaks `import datetime`.
    body = "    import " + ", ".join(sorted(ALLOWED_IMPORTS)) + "\n    return rows"
    result = run_transform(transform_source(body), ROWS)
    assert result.ok, result.error


@pytest.mark.parametrize("module", ["socket", "urllib", "http", "ssl"])
def test_network_modules_are_unavailable(module):
    result = run_transform(transform_source(f"    import {module}\n    return rows"), ROWS)
    assert not result.ok
    assert module in (result.error or "")


@pytest.mark.parametrize("module", ["os", "subprocess", "shutil", "pathlib", "ctypes"])
def test_process_and_filesystem_modules_are_unavailable(module):
    # `os` and `subprocess` are already in `sys.modules` when user code runs, so
    # the import hook alone would not stop them. The driver purges them first.
    result = run_transform(transform_source(f"    import {module}\n    return rows"), ROWS)
    assert not result.ok
    assert module in (result.error or "")


def test_opening_a_file_outside_the_workdir_is_refused():
    result = run_transform(
        transform_source("    open('/etc/hosts').read()\n    return rows"), ROWS
    )
    assert not result.ok
    assert "no filesystem access" in (result.error or "")


def test_a_runaway_transform_is_stopped():
    result = run_transform(
        transform_source("    while True:\n        pass"), ROWS, timeout_seconds=5, cpu_seconds=2
    )
    assert not result.ok
    assert result.rows == []


def test_a_transform_that_raises_reports_the_exception():
    result = run_transform(transform_source("    return 1 / 0"), ROWS)
    assert not result.ok
    assert "ZeroDivisionError" in (result.error or "")


def test_a_missing_entry_point_is_an_error_not_a_crash():
    result = run_transform("VALUE = 1\n", ROWS)
    assert not result.ok
    assert "transform(rows)" in (result.error or "")


def test_a_non_list_return_is_rejected():
    result = run_transform(transform_source("    return {'a': 1}"), ROWS)
    assert not result.ok
    assert "must return a list" in (result.error or "")


def test_more_rows_than_the_cap_are_rejected():
    result = run_transform(
        transform_source("    return [{'text': 'x'}] * 50"), ROWS, max_output_rows=10
    )
    assert not result.ok
    assert "cap" in (result.error or "")


def test_the_child_cannot_see_the_servers_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-never-be-visible")
    # `os` is unavailable inside, which is the point — the only way to observe
    # the environment is to try, and trying is itself blocked.
    result = run_transform(
        transform_source("    import os\n    return [{'k': os.environ.get('OPENROUTER_API_KEY')}]"),
        ROWS,
    )
    assert not result.ok
    assert "sk-should-never-be-visible" not in (result.error or "")
