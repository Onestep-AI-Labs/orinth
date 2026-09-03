"""`orinth doctor` — where am I pointed, and is anything there.

The first question is the one that matters. A CLI silently talking to the wrong
server is the worst failure mode this design has: every command succeeds, the
numbers are real, and they describe someone else's workspace. So `doctor` always
prints the resolved backend and project *with the tier they came from* — "from
ORINTH_BACKEND", "from /Users/x/proj/.orinth.toml" — rather than just the value.
"""

import argparse

from app.cli import __version__, output
from app.cli.commands._common import add_common, make_client, resolve_project
from app.cli.config import find_project_config
from app.cli.errors import EXIT_OK, EXIT_UNREACHABLE, CliError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orinth doctor", allow_abbrev=False)
    add_common(parser)
    return parser


def run(argv: list[str], config, globals_) -> int:
    build_parser().parse_args(argv)
    project = resolve_project(config, globals_)
    found = find_project_config(config.workspace)

    report: dict = {
        "cli_version": __version__,
        "backend": config.backend_url,
        "backend_source": config.backend.source,
        "project": project,
        "project_source": config.project.source,
        "workspace": str(config.workspace),
        "config_file": str(found) if found else None,
        "reachable": False,
    }

    client = make_client(config, globals_)
    try:
        health = client.get("/health") or {}
        report["reachable"] = True
        report["server_version"] = health.get("version")
        report["server_app"] = health.get("app")
        report["projects"] = len(client.get("/api/projects") or [])
        report["models"] = len(client.get("/api/models", params={"project_id": project}) or [])
    except CliError as failure:
        report["error"] = failure.message

    if globals_.json or globals_.jsonl:
        output.emit_json(report)
    else:
        output.key_values(
            [
                ("cli version", report["cli_version"]),
                ("backend", f"{report['backend']}  (from {report['backend_source']})"),
                ("project", f"{report['project']}  (from {report['project_source']})"),
                ("workspace", report["workspace"]),
                ("config file", report["config_file"] or "none found"),
                ("reachable", report["reachable"]),
            ]
            + (
                [
                    ("server version", report.get("server_version")),
                    ("projects", report.get("projects")),
                    ("models", report.get("models")),
                ]
                if report["reachable"]
                else [("error", report.get("error"))]
            )
        )

    if not report["reachable"]:
        # `doctor` reporting a down backend is a successful diagnosis, but the
        # exit code has to say the backend is down or a CI gate cannot use it.
        return EXIT_UNREACHABLE
    if report.get("server_version") and report["server_version"] != __version__:
        output.warn(
            f"CLI is {__version__} but the server is {report['server_version']}."
        )
    return EXIT_OK
