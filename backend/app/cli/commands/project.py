"""`orinth project` — what `--project` can name."""

import argparse

from app.cli import output
from app.cli.commands._common import add_common, emit, make_client
from app.cli.errors import EXIT_OK, EXIT_USAGE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orinth project", allow_abbrev=False)
    subs = parser.add_subparsers(dest="verb")

    add_common(subs.add_parser("ls", help="List projects"))

    show = subs.add_parser("show", help="Show one project's stats")
    show.add_argument("project_id")
    add_common(show)

    create = subs.add_parser("create", help="Create a project")
    create.add_argument("name")
    create.add_argument("--task", action="append", default=[], dest="task_types")
    create.add_argument("--description", default=None)
    add_common(create)

    return parser


def run(argv: list[str], config, globals_) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return EXIT_USAGE
    client = make_client(config, globals_)
    return {"ls": _ls, "show": _show, "create": _create}[args.verb](client, args, globals_)


def _ls(client, args, globals_) -> int:
    rows = client.get("/api/projects") or []
    emit(rows, [("ID", "id"), ("NAME", "name"), ("TASKS", "task_types")], globals_)
    return EXIT_OK


def _show(client, args, globals_) -> int:
    output.emit_json(client.get(f"/api/projects/{args.project_id}/stats"))
    return EXIT_OK


def _create(client, args, globals_) -> int:
    payload = {"name": args.name}
    if args.task_types:
        payload["task_types"] = args.task_types
    if args.description:
        payload["description"] = args.description
    project = client.post("/api/projects", json=payload)
    if globals_.json:
        output.emit_json(project)
    else:
        print(project.get("id"), flush=True)
    return EXIT_OK
