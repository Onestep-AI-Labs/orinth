"""The test that keeps the CLI a client.

`app/cli/` lives inside `app`, whose sibling modules import numpy, PIL, and
SQLAlchemy at module scope, and — one import deeper — TensorFlow and Torch. The
two-phase dispatch in `main.py` exists so `orinth --help` pays for none of it,
and an intention is not a guarantee: the tenth command added by someone who has
not read that docstring will import `app.container` for convenience and nothing
will complain.

So this asserts the property directly. If it fails, the fix is to move the
import inside a handler, not to relax the list.
"""

import subprocess
import sys
import textwrap

import pytest

from app.cli import config as cli_config
from app.cli.errors import EXIT_OK, EXIT_USAGE
from app.cli.main import GROUPS, main

#: Modules whose presence after a dispatch means the laziness is gone. `uvicorn`
#: and `app.main` are on the list even though `serve` legitimately needs them —
#: it imports them inside its handler, so merely *dispatching* must not pull
#: them in.
FORBIDDEN = (
    "tensorflow",
    "torch",
    "ultralytics",
    "cv2",
    "numpy",
    "PIL",
    "sqlalchemy",
    "uvicorn",
)

#: Run in a subprocess because pytest has already imported half the application
#: by the time this file executes; asserting on the in-process `sys.modules`
#: would test the test runner, not the CLI.
#: The commands under test print their own output to stdout, so the probe's
#: findings are tagged rather than positional — `--help` writes a usage block
#: first, and parsing line 0 read that instead of the exit code.
MARKER = "@@probe@@"

PROBE = textwrap.dedent(
    """
    import sys
    from app.cli.main import main
    try:
        code = main({argv!r})
    except SystemExit as stop:
        # argparse hard-exits on `--help` rather than returning. The spec accepts
        # that instead of overriding it, so the probe has to catch it — the
        # question here is what got imported, and that is answerable either way.
        code = stop.code or 0
    leaked = sorted(
        name
        for name in sys.modules
        if name.split(".")[0] in {forbidden!r}
        or name == "app.container"
        or name.startswith("app.services.")
    )
    print("{marker}", code, ",".join(leaked))
    """
)


def _probe(argv: list[str]) -> tuple[int, list[str]]:
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(argv=argv, forbidden=set(FORBIDDEN), marker=MARKER)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    line = next(
        (row for row in result.stdout.splitlines() if row.startswith(MARKER)), None
    )
    assert line is not None, f"probe printed no result:\n{result.stdout}"
    _, code, *rest = line.split(" ", 2)
    leaked = [name for name in (rest[0] if rest else "").split(",") if name]
    return int(code), leaked


@pytest.mark.parametrize("argv", [["--help"], ["--version"], ["dataset", "--help"]])
def test_dispatch_imports_nothing_heavy(argv):
    code, leaked = _probe(argv)
    assert code == EXIT_OK
    assert leaked == [], f"{argv} imported {leaked}"


def test_an_unknown_command_is_answered_from_the_table_without_importing_anything():
    """The group table holds module path *strings* precisely so this is possible."""
    code, leaked = _probe(["nonsense"])
    assert code == EXIT_USAGE
    assert leaked == []


def test_every_group_in_the_table_resolves_to_a_real_module():
    """A typo in the table is otherwise only discovered by a user typing that
    command, since nothing imports it until then."""
    import importlib

    for name, (path, description) in GROUPS.items():
        module = importlib.import_module(path)
        assert hasattr(module, "run"), f"{name} has no run()"
        assert hasattr(module, "build_parser"), f"{name} has no build_parser()"
        assert description, f"{name} has no help text"


def test_serve_defers_its_heavy_imports_to_the_handler():
    """`serve` is the one command allowed to import the server. Importing the
    module must still be free — only calling it may cost."""
    code, leaked = _probe(["serve", "--help"])
    assert code == EXIT_OK
    assert leaked == []


# --- config resolution --------------------------------------------------------


def test_flags_beat_environment_beats_config_beats_default(tmp_path, monkeypatch):
    (tmp_path / ".orinth.toml").write_text(
        'backend = "http://from-file:1"\nproject = "from-file"\n', encoding="utf-8"
    )
    monkeypatch.setattr(cli_config.Path, "home", classmethod(lambda cls: tmp_path.parent))

    from_file = cli_config.resolve(workspace=str(tmp_path), environ={})
    assert from_file.backend.value == "http://from-file:1"
    assert from_file.backend.source.endswith(".orinth.toml")

    from_env = cli_config.resolve(
        workspace=str(tmp_path), environ={"ORINTH_BACKEND": "http://from-env:2"}
    )
    assert from_env.backend.value == "http://from-env:2"
    assert from_env.backend.source == "ORINTH_BACKEND"

    from_flag = cli_config.resolve(
        backend="http://from-flag:3",
        workspace=str(tmp_path),
        environ={"ORINTH_BACKEND": "http://from-env:2"},
    )
    assert from_flag.backend.value == "http://from-flag:3"
    assert from_flag.backend.source == "--backend"


def test_tiers_do_not_merge(tmp_path, monkeypatch):
    """A file supplying only `project` must not inherit a `backend` from it.

    Merging is the tempting behavior and the wrong one: the result is a CLI
    talking to a server nobody chose, which is the worst failure this design has.
    """
    (tmp_path / ".orinth.toml").write_text('project = "only-project"\n', encoding="utf-8")
    monkeypatch.setattr(cli_config.Path, "home", classmethod(lambda cls: tmp_path.parent))

    resolved = cli_config.resolve(workspace=str(tmp_path), environ={})
    assert resolved.project.value == "only-project"
    assert resolved.backend.value == cli_config.DEFAULT_BACKEND
    assert resolved.backend.source == "default"


def test_a_malformed_config_is_a_usage_error_naming_the_file(tmp_path, monkeypatch):
    bad = tmp_path / ".orinth.toml"
    bad.write_text("backend = [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(cli_config.Path, "home", classmethod(lambda cls: tmp_path.parent))

    with pytest.raises(Exception) as failure:
        cli_config.resolve(workspace=str(tmp_path), environ={})
    assert failure.value.code == EXIT_USAGE
    assert str(bad) in failure.value.message


def test_the_completion_script_lists_every_group_the_table_holds():
    """A completion that offers a command the CLI does not have, or omits one it
    does, is worse than none — the user trusts Tab."""
    from app.cli.commands import completion

    assert set(completion.VERBS) == set(GROUPS)


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_prints_a_script_and_exits_clean(shell, capsys):
    from app.cli.commands import completion

    code = completion.run([shell], None, None)
    printed = capsys.readouterr().out

    assert code == EXIT_OK
    assert "orinth" in printed
    assert "dataset" in printed


def test_the_search_stops_at_home(tmp_path, monkeypatch):
    """Walking past `$HOME` would let a stray file in `/` configure every project
    on the machine, and a config nobody remembers writing is worse than none."""
    home = tmp_path / "home"
    nested = home / "a" / "b"
    nested.mkdir(parents=True)
    (tmp_path / ".orinth.toml").write_text('backend = "http://above-home:1"\n', encoding="utf-8")
    monkeypatch.setattr(cli_config.Path, "home", classmethod(lambda cls: home))

    assert cli_config.find_project_config(nested) is None


# --- exit codes ---------------------------------------------------------------


def test_unknown_command_exits_two_not_one():
    """CI branches on these. A usage mistake and a runtime failure are different
    events and must not share a code."""
    assert main(["definitely-not-a-command"]) == EXIT_USAGE


def test_backend_unreachable_exits_four():
    from app.cli.errors import EXIT_UNREACHABLE

    # Port 9 (discard) refuses immediately, so this does not wait on a timeout.
    assert main(["--backend", "http://127.0.0.1:9", "dataset", "ls"]) == EXIT_UNREACHABLE
