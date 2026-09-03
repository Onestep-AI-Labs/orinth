"""`orinth dataset` — the group the whole CLI exists for.

The flow this makes possible in one line is `ingest → prep → readiness`, which
is the only way to get data into Orinth from a terminal or a CI job. Two pieces
carry the weight:

**`relative_paths` on ingest.** Phase 21 established that the directory layout
*is* the detection signal — `data.yaml` beside `train/labels/` means YOLO,
sibling `normal/` and `kista/` folders mean classification with those class
names. A folder uploaded as a flat list of filenames destroys that and degrades
to "a pile of files", so every upload sends each file's path relative to the
folder root alongside it.

**Exit 3 for "not trainable".** Distinct from a runtime failure because CI
branches on it: `orinth dataset readiness $ID` is the gate a pipeline puts in
front of a training step, and "your labels are missing" is a different event from
"the server fell over".
"""

import argparse
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from app.cli import output
from app.cli.commands._common import add_common, emit, make_client, resolve_project
from app.cli.errors import EXIT_FAILURE, EXIT_NOT_READY, EXIT_OK, EXIT_USAGE, CliError
from app.cli.progress import follow, prep_line

#: Skipped unless `--all`. Uploading `.git` is never intended and is often larger
#: than the dataset.
NOISE = {".git", "__MACOSX", ".DS_Store", ".ipynb_checkpoints"}
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")
#: Bounds on client-side archive expansion. The archive is the user's own file,
#: but a zip bomb is still a zip bomb, and failing with a number beats filling
#: the disk.
MAX_MEMBERS = 20_000
MAX_BYTES = 5 * 1024**3
#: Files per multipart request. A 15,000-file folder in one request is a body no
#: proxy will hold; batching keeps each request ordinary and lets progress move.
BATCH = 200

PREP_SETTLED = {"ready", "planned", "failed", "cancelled", "draft"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orinth dataset", allow_abbrev=False)
    subs = parser.add_subparsers(dest="verb", required=True)

    add_common(subs.add_parser("ls", help="List datasets in the project"))

    show = subs.add_parser("show", help="Show one dataset in detail")
    show.add_argument("dataset_id")
    add_common(show)

    ingest = subs.add_parser("ingest", help="Upload files, a folder, or an archive")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--name", default=None)
    ingest.add_argument("--into", default=None, help="Add to an existing dataset id")
    ingest.add_argument("--prep", action="store_true", help="Run prep immediately after upload")
    ingest.add_argument("--batch", type=int, default=BATCH)
    ingest.add_argument("--all", action="store_true", help="Include dotfiles and .git")
    ingest.add_argument("--max-files", type=int, default=MAX_MEMBERS)
    ingest.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    add_common(ingest)

    prep = subs.add_parser("prep", help="Detect, plan, and apply")
    prep.add_argument("dataset_id")
    prep.add_argument("--no-apply", action="store_true", help="Stop after planning")
    prep.add_argument("--detach", action="store_true", help="Queue and return immediately")
    add_common(prep)

    readiness = subs.add_parser("readiness", help="Is this dataset trainable?")
    readiness.add_argument("dataset_id")
    readiness.add_argument("--advisory-fatal", action="store_true")
    add_common(readiness)

    export = subs.add_parser("export", help="Download a dataset as a zip")
    export.add_argument("dataset_id")
    export.add_argument("--to", required=True, help="Destination path, or - for stdout")
    export.add_argument("--splits", default=None, help="Comma-separated subset")
    export.add_argument("--version", dest="version_id", default=None)
    add_common(export)

    remove = subs.add_parser("rm", help="Delete datasets")
    remove.add_argument("dataset_ids", nargs="+")
    remove.add_argument("--yes", action="store_true")
    add_common(remove)

    return parser


def run(argv: list[str], config, globals_) -> int:
    args = build_parser().parse_args(argv)
    client = make_client(config, globals_)
    project = resolve_project(config, globals_)
    handler = {
        "ls": _ls,
        "show": _show,
        "ingest": _ingest,
        "prep": _prep,
        "readiness": _readiness,
        "export": _export,
        "rm": _rm,
    }[args.verb]
    return handler(client, args, globals_, project)


# --- read ---------------------------------------------------------------------


def _ls(client, args, globals_, project) -> int:
    datasets = client.get("/api/datasets", params={"project_id": project}) or []
    rows = [
        {
            **dataset,
            "readiness_state": (dataset.get("readiness") or {}).get("state"),
            "items": sum(
                (split or {}).get("item_count", 0)
                for split in (dataset.get("splits") or {}).values()
            ),
        }
        for dataset in datasets
    ]
    emit(
        rows,
        [
            ("ID", "id"),
            ("NAME", "name"),
            ("TASK", "task_type"),
            ("FORMAT", "format"),
            ("ITEMS", "items"),
            ("READINESS", "readiness_state"),
        ],
        globals_,
    )
    return EXIT_OK


def _show(client, args, globals_, project) -> int:
    dataset = client.get(f"/api/datasets/{args.dataset_id}")
    try:
        plan = client.get(f"/api/datasets/{args.dataset_id}/prep")
    except CliError:
        # A dataset that has never met the prep agent 404s here. That is an
        # answer, not a failure.
        plan = None

    if globals_.json or globals_.jsonl:
        output.emit_json({"dataset": dataset, "plan": plan})
        return EXIT_OK

    readiness = dataset.get("readiness") or {}
    splits = dataset.get("splits") or {}
    output.key_values(
        [
            ("id", dataset.get("id")),
            ("name", dataset.get("name")),
            ("task", dataset.get("task_type")),
            ("format", dataset.get("format")),
            ("labels", dataset.get("labels")),
            ("origin", dataset.get("origin")),
            ("readiness", readiness.get("state")),
            ("trainable", readiness.get("trainable")),
            ("summary", readiness.get("summary")),
        ]
        + [
            (f"split.{name}", (value or {}).get("item_count", 0))
            for name, value in sorted(splits.items())
        ]
        + ([("plan.task", plan.get("task_type")), ("plan.engine", (plan.get("engine") or {}).get("mode"))] if plan else [])
    )
    return EXIT_OK


def _readiness(client, args, globals_, project) -> int:
    readiness = client.get(f"/api/datasets/{args.dataset_id}/readiness")
    if globals_.json or globals_.jsonl:
        output.emit_json(readiness)
    else:
        checks = readiness.get("checks") or []
        output.table(
            [
                [
                    "pass" if check.get("passed") else "FAIL",
                    check.get("severity"),
                    check.get("id"),
                    check.get("label"),
                    check.get("detail") or "",
                ]
                for check in checks
            ],
            ["result", "severity", "id", "label", "detail"],
        )
        output.note(readiness.get("summary") or "")
        if readiness.get("busy"):
            # Phase 21 is explicit that a non-destructive detect/plan leaves the
            # dataset exactly as trainable as it was, so this is reported and
            # deliberately does not move the exit code.
            output.note("Orinth is analysing this dataset; the verdict above is unaffected.")

    if not readiness.get("trainable"):
        return EXIT_NOT_READY
    if args.advisory_fatal and any(
        not check.get("passed") and check.get("severity") == "advisory"
        for check in readiness.get("checks") or []
    ):
        output.error("Advisory checks failed and --advisory-fatal was given.")
        return EXIT_NOT_READY
    return EXIT_OK


# --- ingest -------------------------------------------------------------------


def _collect(paths: list[str], include_all: bool, max_files: int, max_bytes: int, stack) -> list[tuple[Path, str]]:
    """Every file to upload, paired with the relative path the server must see."""
    collected: list[tuple[Path, str]] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise CliError(f"No such file or directory: {path}", code=EXIT_USAGE)
        if path.is_file() and path.name.lower().endswith(ARCHIVE_SUFFIXES):
            root = _expand(path, max_files, max_bytes, stack)
            collected += _walk(root, include_all)
        elif path.is_dir():
            collected += _walk(path, include_all)
        else:
            collected.append((path, path.name))
    if not collected:
        raise CliError("Nothing to upload.", code=EXIT_USAGE)
    if len(collected) > max_files:
        raise CliError(
            f"{len(collected)} files exceeds the {max_files} cap.",
            code=EXIT_USAGE,
            hint="Raise it with --max-files, or ingest a subdirectory at a time.",
        )
    return collected


def _walk(root: Path, include_all: bool) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not include_all and any(
            part in NOISE or part.startswith(".") for part in relative.parts
        ):
            continue
        found.append((path, relative.as_posix()))
    return found


def _expand(archive: Path, max_files: int, max_bytes: int, stack: list) -> Path:
    """Unpack an archive client-side, refusing anything that escapes its root.

    Phase 21 deferred server-side zip ingest for want of an extraction guard.
    Doing it here gets the feature without adding an untrusted-extraction path to
    the server: the archive is the user's own file on the user's own machine, and
    the guard below still refuses absolute paths, `..` segments, and symlinks —
    the three ways an archive writes outside where it was told to.
    """
    target = Path(tempfile.mkdtemp(prefix="orinth-ingest-"))
    stack.append(target)
    total = 0
    name = archive.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            _guard(len(members), max_files)
            for member in members:
                if member.is_dir():
                    continue
                total = _guard_bytes(total + member.file_size, max_bytes)
                _safe_member(member.filename, target)
                bundle.extract(member, target)
    else:
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            _guard(len(members), max_files)
            for member in members:
                if not member.isfile():
                    # A symlink or device node in an archive has no business in
                    # a dataset and is the classic escape vector.
                    if member.issym() or member.islnk():
                        raise CliError(f"Refusing symlink in archive: {member.name}")
                    continue
                total = _guard_bytes(total + member.size, max_bytes)
                _safe_member(member.name, target)
                bundle.extract(member, target)
    return target


def _guard(count: int, cap: int) -> None:
    if count > cap:
        raise CliError(f"Archive holds {count} members, over the {cap} cap.", code=EXIT_USAGE)


def _guard_bytes(total: int, cap: int) -> int:
    if total > cap:
        raise CliError(f"Archive expands past the {cap} byte cap.", code=EXIT_USAGE)
    return total


def _safe_member(name: str, root: Path) -> None:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CliError(f"Refusing archive member outside its root: {name}")
    resolved = (root / candidate).resolve()
    if root.resolve() not in resolved.parents and resolved != root.resolve():
        raise CliError(f"Refusing archive member outside its root: {name}")


def _ingest(client, args, globals_, project) -> int:
    import shutil

    temp_dirs: list[Path] = []
    try:
        files = _collect(args.paths, args.all, args.max_files, args.max_bytes, temp_dirs)
        output.note(f"{len(files)} files to upload")

        dataset_id = args.into
        uploaded = 0
        for index in range(0, len(files), max(1, args.batch)):
            chunk = files[index : index + max(1, args.batch)]
            handles = []
            form: list[tuple[str, tuple[str, object]]] = []
            # httpx encodes a repeated form field as `{key: [v1, v2]}`; a list of
            # `(key, value)` pairs is silently a different thing and fails deep
            # inside h11 rather than at the call site.
            data: dict[str, object] = {"relative_paths": [relative for _, relative in chunk]}
            try:
                for path, _relative in chunk:
                    handle = path.open("rb")
                    handles.append(handle)
                    form.append(("files", (path.name, handle)))
                if dataset_id is None:
                    data["project_id"] = project
                    if args.name:
                        data["name"] = args.name
                    dataset = client.post(
                        "/api/datasets/ingest", files=form, data=data, timeout=3600.0
                    )
                    dataset_id = dataset["id"]
                else:
                    client.post(
                        f"/api/datasets/{dataset_id}/ingest",
                        files=form,
                        data=data,
                        timeout=3600.0,
                    )
            finally:
                for handle in handles:
                    handle.close()
            uploaded += len(chunk)
            output.note(f"staged {uploaded}/{len(files)}")

        if args.prep:
            return _run_prep(client, dataset_id, globals_, auto_apply=True, detach=False)

        dataset = client.get(f"/api/datasets/{dataset_id}")
        if globals_.json or globals_.jsonl:
            output.emit_json(dataset)
        else:
            # The bare id on stdout is what makes `DS=$(orinth dataset ingest .)`
            # work, which is the whole point of a scriptable ingest.
            print(dataset_id, flush=True)
        return EXIT_OK
    finally:
        for directory in temp_dirs:
            shutil.rmtree(directory, ignore_errors=True)


# --- prep ---------------------------------------------------------------------


def _prep(client, args, globals_, project) -> int:
    return _run_prep(
        client,
        args.dataset_id,
        globals_,
        auto_apply=not args.no_apply,
        detach=args.detach,
    )


def _run_prep(client, dataset_id: str, globals_, *, auto_apply: bool, detach: bool) -> int:
    client.post(f"/api/datasets/{dataset_id}/prep", json={"auto_apply": auto_apply})
    if detach:
        print(dataset_id, flush=True)
        return EXIT_OK

    status = follow(
        poll=lambda: client.get(f"/api/datasets/{dataset_id}/prep/status") or {},
        is_done=lambda state: str(state.get("state")) in PREP_SETTLED,
        describe=prep_line,
        reattach=f"orinth dataset show {dataset_id}",
    )

    if status.get("state") == "failed":
        raise CliError(status.get("error") or "Prep failed.")

    dataset = client.get(f"/api/datasets/{dataset_id}")
    try:
        plan = client.get(f"/api/datasets/{dataset_id}/prep")
    except CliError:
        plan = None

    if globals_.json or globals_.jsonl:
        output.emit_json({"dataset": dataset, "plan": plan})
    else:
        print(dataset_id, flush=True)

    readiness = dataset.get("readiness") or {}
    if not readiness.get("trainable"):
        reason = (plan or {}).get("needs_input") or readiness.get("summary") or ""
        output.error("Prepared, but the dataset is still not trainable.", reason)
        return EXIT_NOT_READY
    output.note(readiness.get("summary") or "Ready to train.")
    return EXIT_OK


# --- export / rm --------------------------------------------------------------


def _export(client, args, globals_, project) -> int:
    params = {}
    if args.splits:
        params["splits"] = args.splits
    if args.version_id:
        params["version_id"] = args.version_id
    path = f"/api/datasets/{args.dataset_id}/download"

    if args.to == "-":
        # The single case where stdout carries bytes rather than a table, and it
        # is opt-in precisely because it breaks the pipe-clean rule everywhere
        # else.
        import httpx

        with httpx.stream("GET", f"{client.base_url}{path}", params=params, timeout=3600.0) as response:
            if response.status_code >= 400:
                response.read()
                raise CliError(f"Export failed ({response.status_code})")
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                sys.stdout.buffer.write(chunk)
        return EXIT_OK

    destination = client.download(path, Path(args.to).expanduser(), params=params)
    output.note(f"wrote {destination} ({destination.stat().st_size:,} bytes)")
    return EXIT_OK


def _rm(client, args, globals_, project) -> int:
    if not args.yes:
        if not sys.stdin.isatty():
            # A CI typo must not delete a dataset. Refusing is the only safe
            # default when there is nobody to ask.
            raise CliError(
                "Refusing to delete without --yes when stdin is not a terminal.",
                code=EXIT_USAGE,
            )
        answer = input(f"Delete {len(args.dataset_ids)} dataset(s)? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            output.note("Cancelled.")
            return EXIT_OK

    failures = []
    for dataset_id in args.dataset_ids:
        try:
            client.delete(f"/api/datasets/{dataset_id}")
            output.note(f"deleted {dataset_id}")
        except CliError as failure:
            failures.append((dataset_id, failure.message))

    for dataset_id, message in failures:
        output.error(f"{dataset_id}: {message}")
    return EXIT_FAILURE if failures else EXIT_OK
