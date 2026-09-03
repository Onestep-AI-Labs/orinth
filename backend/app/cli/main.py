"""`orinth` — the entrypoint, and the reason `--help` is fast.

Dispatch is two-phase, and the shape is dictated by one constraint: this package
lives inside `app`, whose sibling modules import numpy, PIL, SQLAlchemy, and —
one import deeper — TensorFlow and Torch. An `argparse` tree that builds every
subparser at startup imports every command module, and `orinth --help` would pay
for all of it.

So:

- The table below maps a group name to a module path **string** and a one-line
  description. `--help`, `version`, and an unknown group are answered from it
  with no import at all.
- Phase 1 parses the global flags and the group token with a small parser.
- Phase 2 imports only that group's module and hands it the rest of the argv.

`app/cli/commands/serve.py` imports uvicorn and `app.main` inside its handler.
That is the single sanctioned heavy import in the package, and it belongs to the
one command whose entire job is to start the server.

This is enforced rather than intended: `tests/test_cli_dispatch.py` asserts that
after `--help` and after a stubbed `dataset ls`, `sys.modules` holds none of
tensorflow, torch, cv2, numpy, PIL, sqlalchemy, fastapi, uvicorn, `app.container`,
or any `app.services.*`. That test is what keeps this true after the tenth
command is added.
"""

import argparse
import importlib
import sys

from app.cli import __version__, output
from app.cli.config import resolve
from app.cli.errors import EXIT_OK, EXIT_USAGE, CliError

#: group -> (module path, one-line help). Values are strings, never modules.
GROUPS: dict[str, tuple[str, str]] = {
    "dataset": ("app.cli.commands.dataset", "List, ingest, prepare, and export datasets"),
    "train": ("app.cli.commands.train", "Start and follow training runs"),
    "test": ("app.cli.commands.test", "Evaluate a model against a dataset"),
    "infer": ("app.cli.commands.infer", "Run inference on a file or a folder"),
    "model": ("app.cli.commands.model", "List, download, and remove models"),
    "project": ("app.cli.commands.project", "List and create projects"),
    "serve": ("app.cli.commands.serve", "Start the Orinth backend"),
    "doctor": ("app.cli.commands.doctor", "Check the environment and the backend"),
    "completion": ("app.cli.commands.completion", "Print a shell completion script"),
}

#: `eval` reads better than `test` for "score this model", and `test` reads
#: better in a CI script. Both exist rather than picking a side.
ALIASES = {"eval": "test"}

USAGE = "orinth [--backend URL] [--project ID] <command> [...]"


def _global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orinth", usage=USAGE, add_help=False, allow_abbrev=False
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--wait-backend", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--version", action="store_true")
    return parser


def _print_help() -> None:
    width = max(len(name) for name in GROUPS)
    lines = [f"usage: {USAGE}", "", "commands:"]
    lines += [f"  {name.ljust(width)}  {help_text}" for name, (_, help_text) in GROUPS.items()]
    lines += [
        "",
        "global options:",
        "  --backend URL      Orinth backend to talk to (env ORINTH_BACKEND)",
        "  --project ID       project to operate in (env ORINTH_PROJECT)",
        "  --workspace DIR    where to look for .orinth.toml",
        "  --json / --jsonl   machine-readable output on stdout",
        "  --no-header        omit table headers",
        "  --no-color         disable colour on stderr",
        "  -q / -v            quieter / more verbose progress on stderr",
        "  --timeout SECONDS  HTTP read timeout (default 60)",
        "  --wait-backend N   wait up to N seconds for the backend to answer",
        "",
        "Run `orinth <command> --help` for a command's options.",
    ]
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _global_parser()
    globals_, rest = parser.parse_known_args(argv)

    output.configure(
        quiet=globals_.quiet,
        # NO_COLOR is the cross-tool convention and costs one lookup to honour.
        color=not globals_.no_color and "NO_COLOR" not in _environ(),
        header=not globals_.no_header,
    )

    if globals_.version:
        print(__version__)
        return EXIT_OK

    group = rest[0] if rest else None
    if group is None:
        _print_help()
        return EXIT_OK

    # `parse_known_args` above consumed `--help`, so it has to be handed back to
    # the group parser or `orinth serve --help` reaches serve's handler with no
    # arguments and starts a server. Found exactly that way.
    if globals_.help:
        rest = [*rest, "--help"]

    group = ALIASES.get(group, group)
    if group not in GROUPS:
        output.error(
            f"Unknown command '{rest[0]}'.",
            hint=f"Try one of: {', '.join(GROUPS)}",
        )
        return EXIT_USAGE

    try:
        config = resolve(
            backend=globals_.backend, project=globals_.project, workspace=globals_.workspace
        )
        if globals_.verbose:
            output.note(f"backend {config.backend_url} (from {config.backend.source})")
            output.note(f"project {config.project.value} (from {config.project.source})")

        module = importlib.import_module(GROUPS[group][0])
        return int(module.run(rest[1:], config, globals_))
    except CliError as failure:
        output.error(failure.message, failure.hint)
        return failure.code
    except KeyboardInterrupt:
        # A second Ctrl-C during a follow, or one outside a follow. Either way
        # the shell convention is 130 and no traceback.
        output.note("")
        return 130
    except BrokenPipeError:
        return EXIT_OK


def _environ() -> dict:
    import os

    return os.environ


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
