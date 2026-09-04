"""`orinth train` — start a run, and refuse to start a doomed one.

**Preflight is the headline feature, not a nicety.** Phase 21's motivating bug is
that `keras_common.load_split` silently skips images whose annotation file is
missing, so an unlabelled classification dataset trains on zero items and reports
success. The readiness contract exists to catch exactly that, and a CLI that
posts a job without consulting it would reintroduce the bug for every terminal
user. So `train` reads the dataset first and exits 3 before creating anything.

`--set k=v` is how every phase-13 and phase-14 advanced parameter is reachable
without a flag per knob. Values go through `json.loads` first so `--set lora_r=32`
is an int and `--set packing=true` is a bool, falling back to the raw string,
which is what makes `--set finetune_method=qlora` work too.
"""

import argparse
import json

from app.cli import output
from app.cli.commands._common import add_common, emit, make_client, resolve_project
from app.cli.errors import EXIT_FAILURE, EXIT_NOT_READY, EXIT_OK, EXIT_USAGE, CliError
from app.cli.progress import follow, job_line

SETTLED = {"completed", "failed", "canceled", "cancelled"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orinth train", allow_abbrev=False)
    subs = parser.add_subparsers(dest="verb")

    models = subs.add_parser("models", help="List model options for a task")
    models.add_argument("--task", required=True)
    add_common(models)

    listing = subs.add_parser("ls", help="List training jobs")
    add_common(listing)

    show = subs.add_parser("show", help="Show one training job")
    show.add_argument("job_id")
    add_common(show)

    logs = subs.add_parser("logs", help="Print a job's log tail")
    logs.add_argument("job_id")
    logs.add_argument("--follow", action="store_true")
    add_common(logs)

    cancel = subs.add_parser("cancel", help="Cancel a running job")
    cancel.add_argument("job_id")
    add_common(cancel)

    run = subs.add_parser("run", help="Start a training run")
    _run_flags(run)

    return parser


def _run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset_id")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", dest="model_option_id", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V")
    parser.add_argument("--force", action="store_true", help="Skip the readiness preflight")
    parser.add_argument("--detach", action="store_true")
    add_common(parser)


def run(argv: list[str], config, globals_) -> int:
    parser = build_parser()
    # `orinth train <dataset>` with no verb means `run`, which is the form
    # anyone types first. Detecting it by "the first token is not a verb" keeps
    # `run` explicit for scripts without demanding it interactively.
    verbs = {"models", "ls", "show", "logs", "cancel", "run"}
    if argv and argv[0] not in verbs and not argv[0].startswith("-"):
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return EXIT_USAGE

    client = make_client(config, globals_)
    project = resolve_project(config, globals_)
    return {
        "models": _models,
        "ls": _ls,
        "show": _show,
        "logs": _logs,
        "cancel": _cancel,
        "run": _start,
    }[args.verb](client, args, globals_, project)


def _models(client, args, globals_, project) -> int:
    options = client.get("/api/training/model-options", params={"task_type": args.task}) or []
    rows = options if isinstance(options, list) else options.get("options", [])
    # `runnable`, not `available` — the schema field is the former, and asking
    # for the latter printed an empty column on every row.
    emit(
        rows,
        [
            ("ID", "id"),
            ("NAME", "name"),
            ("FAMILY", "family"),
            ("RUNNABLE", "runnable"),
            ("DOWNLOAD", "needs_download"),
        ],
        globals_,
    )
    return EXIT_OK


def _ls(client, args, globals_, project) -> int:
    jobs = client.get("/api/training/jobs", params={"project_id": project}) or []
    rows = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
    emit(
        rows,
        [
            ("ID", "id"),
            ("NAME", "name"),
            ("TASK", "task_type"),
            ("STATUS", "status"),
            ("CREATED", "created_at"),
        ],
        globals_,
    )
    return EXIT_OK


def _show(client, args, globals_, project) -> int:
    job = client.get(f"/api/training/jobs/{args.job_id}")
    if globals_.json or globals_.jsonl:
        output.emit_json(job)
        return EXIT_OK
    output.key_values(
        [(key, job.get(key)) for key in ("id", "name", "task_type", "status", "created_at", "error")]
    )
    return EXIT_OK


def _logs(client, args, globals_, project) -> int:
    def tail() -> dict:
        return client.get(f"/api/training/jobs/{args.job_id}") or {}

    if not args.follow:
        job = tail()
        for line in job.get("logs") or []:
            print(line, flush=True)
        return EXIT_OK

    seen = 0

    def describe(job: dict) -> str:
        nonlocal seen
        lines = job.get("logs") or []
        # Only the suffix that is new, so following does not reprint the log on
        # every poll.
        fresh = lines[seen:]
        seen = len(lines)
        return "\n".join(fresh) if fresh else ""

    follow(
        poll=tail,
        is_done=lambda job: str(job.get("status")) in SETTLED,
        describe=describe,
        reattach=f"orinth train logs {args.job_id} --follow",
    )
    return EXIT_OK


def _cancel(client, args, globals_, project) -> int:
    client.post(f"/api/training/jobs/{args.job_id}/cancel")
    output.note(f"cancel requested for {args.job_id}")
    return EXIT_OK


#: Fields `TrainingJobCreate` declares at the top level. Putting them in
#: `hyperparameters` instead is silently ignored — the model validates, the job
#: starts, and it trains for the default 50 epochs no matter what was asked.
#: Found by running it: `--epochs 1` produced a 50-epoch run.
TOP_LEVEL = {
    "epochs",
    "image_size",
    "batch_size",
    "learning_rate",
    "device",
    "optimizer",
    "patience",
    "workers",
    "cache",
    "base_model",
}


def _overrides(args) -> dict:
    """Parse `--set k=v`, keeping JSON types where the value is JSON."""
    parsed: dict = {}
    for override in args.overrides:
        if "=" not in override:
            raise CliError(f"--set expects k=v, got '{override}'", code=EXIT_USAGE)
        key, _, raw = override.partition("=")
        try:
            parsed[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            # A bare word is a string, which is what `--set finetune_method=qlora` means.
            parsed[key.strip()] = raw
    return parsed


def _split_params(args) -> tuple[dict, dict]:
    """Separate what the schema declares from what rides in `hyperparameters`.

    `--set` reaches both: a key the schema declares goes to the top level where
    the runner reads it, and everything else goes to `hyperparameters`, which is
    how the phase-13 and phase-14 advanced parameters are addressed without a
    flag per knob.
    """
    top: dict = {}
    for key, value in (
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("image_size", args.image_size),
        ("learning_rate", args.lr),
        ("device", args.device),
    ):
        if value is not None:
            top[key] = value

    advanced: dict = {}
    for key, value in _overrides(args).items():
        if key in TOP_LEVEL:
            top[key] = value
        else:
            advanced[key] = value
    return top, advanced


def _start(client, args, globals_, project) -> int:
    if not args.force:
        dataset = client.get(f"/api/datasets/{args.dataset_id}")
        readiness = dataset.get("readiness") or {}
        if not readiness.get("trainable"):
            output.error(
                f"{args.dataset_id} is not ready to train.",
                readiness.get("summary") or "",
            )
            for check in readiness.get("checks") or []:
                if not check.get("passed") and check.get("severity") == "blocking":
                    output.note(f"  FAIL {check.get('id')}: {check.get('detail')}")
            output.note("Run `orinth dataset prep` first, or pass --force to start anyway.")
            return EXIT_NOT_READY
    else:
        output.warn("--force: skipping the readiness preflight entirely.")

    options = client.get("/api/training/model-options", params={"task_type": args.task}) or []
    rows = options if isinstance(options, list) else options.get("options", [])
    valid = {str(option.get("id")) for option in rows if isinstance(option, dict)}
    if valid and args.model_option_id not in valid:
        raise CliError(
            f"Unknown model option '{args.model_option_id}' for task '{args.task}'.",
            code=EXIT_USAGE,
            hint="Valid ids: " + ", ".join(sorted(valid)),
        )

    top, advanced = _split_params(args)
    payload = {
        "project_id": project,
        "dataset_id": args.dataset_id,
        "task_type": args.task,
        "model_option_id": args.model_option_id,
        "hyperparameters": advanced,
        **top,
    }
    # The family comes from the option rather than the schema default, which is
    # `yolo` — starting a Keras run under a yolo family is how a job ends up in
    # the wrong command builder.
    family = next(
        (
            str(option.get("family"))
            for option in rows
            if isinstance(option, dict) and option.get("id") == args.model_option_id
        ),
        None,
    )
    if family:
        payload["model_family"] = family
    if args.name:
        payload["model_name"] = args.name

    job = client.post("/api/training/jobs", json=payload)
    job_id = job.get("id")
    output.note(f"started {job_id}")

    if args.detach:
        print(job_id, flush=True)
        return EXIT_OK

    final = follow(
        poll=lambda: client.get(f"/api/training/jobs/{job_id}") or {},
        is_done=lambda state: str(state.get("status")) in SETTLED,
        describe=job_line,
        reattach=f"orinth train show {job_id}",
    )

    if globals_.json or globals_.jsonl:
        output.emit_json(final)
    else:
        print(job_id, flush=True)

    if str(final.get("status")) != "completed":
        output.error(str(final.get("error") or f"Job finished {final.get('status')}."))
        for line in (final.get("logs") or [])[-20:]:
            output.note(line)
        return EXIT_FAILURE
    return EXIT_OK
