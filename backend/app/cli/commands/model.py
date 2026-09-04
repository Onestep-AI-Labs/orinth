"""`orinth model` — the trained-artifact catalog."""

import argparse
from pathlib import Path

from app.cli import output
from app.cli.commands._common import add_common, emit, make_client, resolve_project
from app.cli.errors import EXIT_OK, EXIT_USAGE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orinth model", allow_abbrev=False)
    subs = parser.add_subparsers(dest="verb")

    add_common(subs.add_parser("ls", help="List models"))

    show = subs.add_parser("show", help="Show one model")
    show.add_argument("model_id")
    add_common(show)

    download = subs.add_parser("download", help="Download a model artifact")
    download.add_argument("model_id")
    download.add_argument("--to", required=True)
    add_common(download)

    remove = subs.add_parser("rm", help="Delete a model")
    remove.add_argument("model_id")
    add_common(remove)

    return parser


def run(argv: list[str], config, globals_) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return EXIT_USAGE
    client = make_client(config, globals_)
    project = resolve_project(config, globals_)
    return {"ls": _ls, "show": _show, "download": _download, "rm": _rm}[args.verb](
        client, args, globals_, project
    )


def _ls(client, args, globals_, project) -> int:
    rows = client.get("/api/models", params={"project_id": project}) or []
    emit(
        rows,
        [("ID", "id"), ("NAME", "name"), ("TASK", "task_type"), ("FAMILY", "family"), ("SOURCE", "source")],
        globals_,
    )
    return EXIT_OK


def _show(client, args, globals_, project) -> int:
    rows = client.get("/api/models", params={"project_id": project}) or []
    match = next((row for row in rows if row.get("id") == args.model_id), None)
    if match is None:
        output.error(f"No model '{args.model_id}'.")
        return 1
    output.emit_json(match) if globals_.json else output.key_values(sorted(match.items()))
    return EXIT_OK


def _download(client, args, globals_, project) -> int:
    destination = client.download(
        f"/api/models/{args.model_id}/download", Path(args.to).expanduser()
    )
    output.note(f"wrote {destination} ({destination.stat().st_size:,} bytes)")
    return EXIT_OK


def _rm(client, args, globals_, project) -> int:
    client.delete(f"/api/models/{args.model_id}")
    output.note(f"deleted {args.model_id}")
    return EXIT_OK
