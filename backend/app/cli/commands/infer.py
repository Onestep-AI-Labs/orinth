"""`orinth infer` — one file or a folder, through the same route the studio uses.

Batch is a client-side loop rather than a server-side batch endpoint, because
`POST /api/inference` is per-file and adding a batch route would duplicate the
per-file semantics (overlay writing, per-model parameter defaults) for no gain.
The loop keeps going after a failure and reports the tally, since one unreadable
image in a folder of 500 is not a reason to lose the other 499.
"""

import argparse
from pathlib import Path

from app.cli import output
from app.cli.commands._common import add_common, emit, make_client, resolve_project
from app.cli.errors import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, CliError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orinth infer", allow_abbrev=False)
    subs = parser.add_subparsers(dest="verb")

    add_common(subs.add_parser("ls", help="List past inference results"))

    show = subs.add_parser("show", help="Show one inference result")
    show.add_argument("inference_id")
    add_common(show)

    remove = subs.add_parser("rm", help="Delete an inference result")
    remove.add_argument("inference_id")
    add_common(remove)

    run = subs.add_parser("run", help="Run inference")
    run.add_argument("model_id")
    run.add_argument("inputs", nargs="*", help="Files or folders; omit with --text")
    run.add_argument("--text", default=None, help="Text input instead of a file")
    run.add_argument("--question", default=None)
    run.add_argument("--confidence", type=float, default=0.65)
    run.add_argument("--iou", type=float, default=0.7)
    run.add_argument("--max-length", type=int, default=120)
    add_common(run)

    return parser


def run(argv: list[str], config, globals_) -> int:
    verbs = {"ls", "show", "rm", "run"}
    if argv and argv[0] not in verbs and not argv[0].startswith("-"):
        argv = ["run", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return EXIT_USAGE

    client = make_client(config, globals_)
    project = resolve_project(config, globals_)
    return {"ls": _ls, "show": _show, "rm": _rm, "run": _start}[args.verb](
        client, args, globals_, project
    )


def _ls(client, args, globals_, project) -> int:
    rows = client.get("/api/inference", params={"project_id": project}) or []
    emit(
        rows,
        [("ID", "id"), ("MODEL", "model_id"), ("FILE", "filename"), ("CREATED", "created_at")],
        globals_,
    )
    return EXIT_OK


def _show(client, args, globals_, project) -> int:
    output.emit_json(client.get(f"/api/inference/{args.inference_id}"))
    return EXIT_OK


def _rm(client, args, globals_, project) -> int:
    client.delete(f"/api/inference/{args.inference_id}")
    output.note(f"deleted {args.inference_id}")
    return EXIT_OK


def _expand(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise CliError(f"No such file or directory: {path}", code=EXIT_USAGE)
        if path.is_dir():
            files += sorted(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            files.append(path)
    return files


def _start(client, args, globals_, project) -> int:
    common = {
        "model_id": args.model_id,
        "project_id": project,
        "confidence_threshold": str(args.confidence),
        "iou_threshold": str(args.iou),
        "max_length": str(args.max_length),
    }
    if args.question:
        common["question"] = args.question

    if args.text is not None:
        result = client.post("/api/inference", data={**common, "text_content": args.text})
        output.emit_json(result)
        return EXIT_OK

    files = _expand(args.inputs)
    if not files:
        raise CliError("Nothing to run on. Pass a file, a folder, or --text.", code=EXIT_USAGE)

    results, failures = [], []
    for index, path in enumerate(files, start=1):
        try:
            with path.open("rb") as handle:
                results.append(
                    client.post(
                        "/api/inference",
                        data=dict(common),
                        files={"file": (path.name, handle)},
                        timeout=600.0,
                    )
                )
        except CliError as failure:
            failures.append((path, failure.message))
        if len(files) > 1:
            output.note(f"{index}/{len(files)} {path.name}")

    output.emit_json(results if len(files) > 1 else (results[0] if results else None))
    for path, message in failures:
        output.error(f"{path}: {message}")
    if len(files) > 1:
        output.note(f"{len(results)} ok, {len(failures)} failed")
    return EXIT_FAILURE if failures else EXIT_OK
