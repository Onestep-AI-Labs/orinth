"""`orinth test` (alias `eval`) — score a model, and let CI fail on the number.

`--fail-under metric=value` is the reason this is worth a command rather than a
`curl`. It is repeatable, and a metric the job did not report exits **2**, not 0:
silently passing a gate that never ran is the failure mode a threshold flag
exists to prevent, and it is invisible in a green pipeline.
"""

import argparse

from app.cli import output
from app.cli.commands._common import add_common, emit, make_client, resolve_project
from app.cli.errors import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, CliError
from app.cli.progress import follow, job_line

SETTLED = {"completed", "failed", "canceled", "cancelled"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orinth test", allow_abbrev=False)
    subs = parser.add_subparsers(dest="verb")

    add_common(subs.add_parser("datasets", help="List evaluation datasets"))
    add_common(subs.add_parser("ls", help="List evaluation jobs"))

    show = subs.add_parser("show", help="Show one evaluation job")
    show.add_argument("job_id")
    add_common(show)

    per_item = subs.add_parser("per-item", help="Per-item rows for a job")
    per_item.add_argument("job_id")
    add_common(per_item)

    compare = subs.add_parser("compare", help="Compare models on one dataset")
    compare.add_argument("job_id")
    add_common(compare)

    run = subs.add_parser("run", help="Evaluate a model")
    run.add_argument("model_id")
    run.add_argument("--dataset", dest="dataset_key", required=True)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--fail-under", action="append", default=[], metavar="METRIC=VALUE")
    run.add_argument("--detach", action="store_true")
    add_common(run)

    return parser


def run(argv: list[str], config, globals_) -> int:
    verbs = {"datasets", "ls", "show", "per-item", "compare", "run"}
    if argv and argv[0] not in verbs and not argv[0].startswith("-"):
        argv = ["run", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return EXIT_USAGE

    client = make_client(config, globals_)
    project = resolve_project(config, globals_)
    return {
        "datasets": _datasets,
        "ls": _ls,
        "show": _show,
        "per-item": _per_item,
        "compare": _compare,
        "run": _start,
    }[args.verb](client, args, globals_, project)


def _datasets(client, args, globals_, project) -> int:
    rows = client.get("/api/testing/datasets") or []
    emit(rows, [("KEY", "key"), ("NAME", "name"), ("TASK", "task_type"), ("ITEMS", "item_count")], globals_)
    return EXIT_OK


def _ls(client, args, globals_, project) -> int:
    rows = client.get("/api/testing/jobs", params={"project_id": project}) or []
    emit(
        rows,
        [("ID", "id"), ("MODEL", "model_id"), ("DATASET", "dataset_key"), ("STATUS", "status")],
        globals_,
    )
    return EXIT_OK


def _show(client, args, globals_, project) -> int:
    job = client.get(f"/api/testing/jobs/{args.job_id}")
    if globals_.json or globals_.jsonl:
        output.emit_json(job)
        return EXIT_OK
    output.key_values(
        [(key, job.get(key)) for key in ("id", "model_id", "dataset_key", "status", "error")]
    )
    metrics = job.get("metrics") or {}
    if metrics:
        output.key_values([(f"metric.{key}", value) for key, value in sorted(metrics.items())])
    return EXIT_OK


def _per_item(client, args, globals_, project) -> int:
    rows = client.get(f"/api/testing/jobs/{args.job_id}/per-image") or []
    if globals_.json or globals_.jsonl:
        output.emit_json(rows)
        return EXIT_OK
    if not rows:
        output.note("No per-item rows for this job.")
        return EXIT_OK
    keys = [key for key in rows[0] if not isinstance(rows[0][key], (dict, list))]
    emit(rows, [(key.upper(), key) for key in keys], globals_)
    return EXIT_OK


def _compare(client, args, globals_, project) -> int:
    comparison = client.get(f"/api/testing/jobs/{args.job_id}/comparison")
    output.emit_json(comparison)
    return EXIT_OK


def _thresholds(raw: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for entry in raw:
        if "=" not in entry:
            raise CliError(f"--fail-under expects metric=value, got '{entry}'", code=EXIT_USAGE)
        key, _, value = entry.partition("=")
        try:
            parsed[key.strip()] = float(value)
        except ValueError as error:
            raise CliError(f"--fail-under value must be a number: '{value}'", code=EXIT_USAGE) from error
    return parsed


def _start(client, args, globals_, project) -> int:
    thresholds = _thresholds(args.fail_under)
    payload = {
        "project_id": project,
        "model_id": args.model_id,
        "dataset_key": args.dataset_key,
    }
    if args.limit is not None:
        payload["limit"] = args.limit

    job = client.post("/api/testing/jobs", json=payload)
    job_id = job.get("id")
    output.note(f"started {job_id}")

    if args.detach:
        print(job_id, flush=True)
        return EXIT_OK

    final = follow(
        poll=lambda: client.get(f"/api/testing/jobs/{job_id}") or {},
        is_done=lambda state: str(state.get("status")) in SETTLED,
        describe=job_line,
        reattach=f"orinth test show {job_id}",
    )

    metrics = final.get("metrics") or {}
    if globals_.json or globals_.jsonl:
        output.emit_json(final)
    else:
        output.key_values(sorted(metrics.items()) or [("metrics", "none reported")])

    if str(final.get("status")) != "completed":
        output.error(str(final.get("error") or f"Job finished {final.get('status')}."))
        return EXIT_FAILURE

    missing = [name for name in thresholds if name not in metrics]
    if missing:
        raise CliError(
            "Threshold names a metric this job did not report: " + ", ".join(missing),
            code=EXIT_USAGE,
            hint="Reported metrics: " + (", ".join(sorted(metrics)) or "none"),
        )

    failed = [
        f"{name}={metrics[name]} < {floor}"
        for name, floor in thresholds.items()
        if float(metrics[name]) < floor
    ]
    if failed:
        output.error("Below threshold: " + "; ".join(failed))
        return EXIT_FAILURE
    return EXIT_OK
