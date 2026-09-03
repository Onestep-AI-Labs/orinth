"""`orinth test` (alias `eval`) — score a model, and let CI fail on the number.

`--fail-under metric=value` is the reason this is worth a command rather than a
`curl`. It is repeatable, and a metric the job did not report exits **2**, not 0:
silently passing a gate that never ran is the failure mode a threshold flag
exists to prevent, and it is invisible in a green pipeline.
"""

import argparse
from typing import Any

from app.cli import output
from app.cli.commands._common import add_common, emit, make_client, resolve_project
from app.cli.errors import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, CliError
from app.cli.progress import follow, job_line

SETTLED = {"completed", "failed", "canceled", "cancelled"}

#: Leaf names worth printing without `--json`. The full tree carries ROC curves
#: and confusion matrices, which are hundreds of numbers and not a summary.
HEADLINE = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
    "mcc",
    "macro_auc",
    "micro_auc",
    "map50",
    "map5095",
    "precision",
    "recall",
    "dice",
    "iou",
    "perplexity",
    "loss",
    "samples",
}


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
    """Models down the rows, metrics across the columns.

    Emitting the raw comparison payload was a placeholder: comparing two models
    is the one thing here a person does by eye, and a nested JSON document is
    the worst possible shape for that. The column set is derived from the
    metrics actually present rather than hardcoded, because it differs by task —
    a detection job reports mAP where a classifier reports F1.
    """
    comparison = client.get(f"/api/testing/jobs/{args.job_id}/comparison")
    if globals_.json or globals_.jsonl:
        output.emit_json(comparison)
        return EXIT_OK

    entries = comparison.get("models") or comparison.get("jobs") or []
    if not isinstance(entries, list) or not entries:
        output.note("Nothing to compare — this job has no sibling runs.")
        return EXIT_OK

    metric_names: list[str] = []
    for entry in entries:
        for name in (entry.get("metrics") or {}):
            if name not in metric_names:
                metric_names.append(name)

    rows = []
    for entry in entries:
        metrics = entry.get("metrics") or {}
        row = [entry.get("model_id") or entry.get("id") or "?", entry.get("status", "")]
        for name in metric_names:
            value = metrics.get(name)
            # Four places is enough to separate two models and short enough to
            # keep the row scannable; `-` for a metric this model did not report
            # is honest where `0.0000` would be a lie.
            row.append(f"{float(value):.4f}" if isinstance(value, (int, float)) else "-")
        rows.append(row)

    output.table(rows, ["model", "status", *metric_names])
    return EXIT_OK


def flatten_metrics(payload: Any, prefix: str = "") -> dict[str, float]:
    """Every scalar in the metrics tree, keyed by its dotted path.

    Evaluation metrics are nested — `image.overall.accuracy`,
    `classification.micro_auc` — so a flat `metrics.get("accuracy")` finds
    nothing and `--fail-under accuracy=0.8` could never match anything. Found by
    running it against a real job: the top-level keys are `classification`,
    `image`, `labels`, `samples`, none of which is a number.

    Booleans are excluded deliberately. `scores: true` is a flag, and a
    threshold against `True == 1.0` would pass silently and mean nothing.
    """
    found: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.update(flatten_metrics(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        found[prefix] = float(payload)
    return found


def resolve_metric(name: str, flat: dict[str, float]) -> tuple[str | None, list[str]]:
    """Find a metric by dotted path, or by leaf name when that is unambiguous.

    Requiring the full path for everything would make the common case
    (`accuracy`) unnecessarily long; accepting a leaf that matches several paths
    would silently gate on whichever happened to sort first. So an ambiguous
    leaf is an error that lists the candidates.
    """
    if name in flat:
        return name, []
    matches = sorted(path for path in flat if path.split(".")[-1] == name)
    if len(matches) == 1:
        return matches[0], []
    return None, matches


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

    flat = flatten_metrics(final.get("metrics") or {})
    if globals_.json or globals_.jsonl:
        output.emit_json(final)
    elif flat:
        # Scalars only, and only the headline ones by default. Printing the raw
        # tree put ROC curves and full confusion matrices on the terminal.
        # Depth matters as much as the name: `precision` is a headline metric
        # at `image.overall.precision` and per-class noise at
        # `image.report.metal.precision`. Two dots is the boundary between a
        # summary and a breakdown.
        headline = {
            path: value
            for path, value in flat.items()
            if path.count(".") <= 1
            or (path.count(".") == 2 and path.split(".")[-1] in HEADLINE)
        }
        output.key_values(sorted((headline or flat).items()))
        if len(flat) > len(headline) and headline:
            output.note(
                f"{len(flat) - len(headline)} more metrics — use --json for the full tree."
            )
    else:
        output.note("This job reported no numeric metrics.")

    if str(final.get("status")) != "completed":
        output.error(str(final.get("error") or f"Job finished {final.get('status')}."))
        return EXIT_FAILURE

    resolved: dict[str, tuple[str, float]] = {}
    for name, floor in thresholds.items():
        path, ambiguous = resolve_metric(name, flat)
        if path is None:
            if ambiguous:
                raise CliError(
                    f"`{name}` matches several metrics; name one exactly.",
                    code=EXIT_USAGE,
                    hint="Candidates: " + ", ".join(ambiguous),
                )
            # Silently passing a gate that never ran is the failure mode a
            # threshold flag exists to prevent, so this is a usage error rather
            # than a pass.
            raise CliError(
                f"Threshold names a metric this job did not report: {name}",
                code=EXIT_USAGE,
                hint="Reported: " + (", ".join(sorted(flat)[:12]) or "none"),
            )
        resolved[name] = (path, floor)

    failed = [
        f"{path}={flat[path]:.4f} < {floor}"
        for path, floor in resolved.values()
        if flat[path] < floor
    ]
    if failed:
        output.error("Below threshold: " + "; ".join(failed))
        return EXIT_FAILURE
    for path, floor in resolved.values():
        output.note(f"ok  {path}={flat[path]:.4f} >= {floor}")
    return EXIT_OK
