# Spec: Phase 23 Command-Line Interface

## Status

**Implemented, partially.** The transport decision, the packaging, the two-phase lazy dispatch, the
config precedence, the output contract, the exit-code table, and the three additive routes are all
in and exercised against a live backend over the 13 datasets in a real workspace.

Landed:

- **Core** — `app/cli/{__init__,main,config,client,output,progress,errors}.py` and
  `commands/_common.py`. `[project.scripts] orinth = "app.cli.main:main"` in `backend/pyproject.toml`.
- **Commands** — `dataset` (ls, show, ingest, prep, readiness, export, rm), `train`
  (run, models, ls, show, logs, cancel), `test`/`eval` (run, datasets, ls, show, per-item, compare),
  `infer` (run, ls, show, rm), `model` (ls, show, download, rm), `project` (ls, show, create),
  `serve`, `doctor`.
- **Routes** — `GET /api/datasets/{id}`, `GET /api/datasets/{id}/download` (backed by a new
  `VersioningMixin.archive_dataset`), and `version`/`app` on `/health`. All three additive.
- **Laziness, enforced** — `tests/test_cli_dispatch.py` runs `main()` in a subprocess and asserts
  `sys.modules` afterwards holds none of tensorflow, torch, ultralytics, cv2, numpy, PIL, sqlalchemy,
  uvicorn, `app.container`, or any `app.services.*` — for `--help`, `--version`, `dataset --help`,
  `serve --help`, and an unknown group. 12 tests covering that plus config precedence and exit codes.

Verified live (server on `:8123`, real workspace):

```
$ orinth dataset ingest /tmp/cli-demo --name "cli demo" --prep
16 files to upload
staged 16/16
[detecting] 10% Reading the files you uploaded
[planning] 35% Read 16 files — looks like classification, checking
[done] 100% Prepared 16 items as classification
cli-demo-314fada8
Ready to train.
```

The class folders (`cardboard/`, `plastic/`) became the dataset's labels, which is the
`relative_paths` requirement working. Exit codes were checked one at a time and all six match the
table below: 0 ready, 3 not-trainable, 1 missing dataset, 2 unknown command, 2 `rm` without `--yes`
on a non-TTY, 4 backend down. `dataset export` produced a 49-entry zip with `_staging/`, `_derived/`,
and `prep_undo.json` correctly excluded. `dataset ls --json | python -m json.tool` parses, so the
stdout/stderr split holds.

Two defects were found and fixed during that run, both worth recording because neither is visible
from reading the code:

- `parse_known_args` consumed `--help` before the group parser saw it, so `orinth serve --help`
  reached serve's handler with no arguments **and started a server**. `--help` is now handed back to
  the group.
- httpx encodes a repeated multipart field as `{key: [v1, v2]}`; a list of `(key, value)` pairs is
  silently a different thing and fails deep inside h11 rather than at the call site. Ingest built
  the wrong one, so no upload worked at all until it was fixed.

**Deferred, and not attempted:** `train --set` and `test --fail-under` are implemented but have not
been run against a real training job (that needs GPU time and a long run); `infer` batch mode is
implemented but only smoke-tested for argument handling; `--wait` on a busy prep run, `--advisory-fatal`,
and `export --version` are coded against the spec but unexercised. Shell completion, the thin
`orinth-cli` wheel, and `test compare`'s table rendering (it currently emits JSON) remain open. Every
number above is measured; nothing in this section is projected.

## Goal

Make the whole platform drivable from a terminal and from CI: `orinth dataset ingest` → `orinth
dataset prep` → `orinth train` → `orinth test` → `orinth infer`, with human-readable output by
default, `--json` everywhere, and exit codes a CI job can branch on.

Today every one of those flows exists only behind the Next.js UI. A user with a folder of images on
a build machine, or a pipeline that wants to fail a PR when a dataset stops being trainable, has no
entry point at all. The web app also hides the single most CI-relevant fact the platform computes —
`DatasetReadiness.trainable`, added in phase 21 — behind a badge on a React page.

The CLI is not a second implementation of the platform. It is a client of the API the UI already
calls, plus the argument parsing, table rendering, and job following that a terminal needs and a
browser does not.

## Scope

In:

- A console script `orinth`, declared in `backend/pyproject.toml`, implemented under
  `backend/app/cli/`.
- Command groups: `dataset`, `train`, `test` (alias `eval`), `infer`, `model`, `project`, `serve`,
  `doctor`, `version`.
- An output contract: aligned tables on a TTY, tab-separated off it, `--json` on every command that
  returns data, data on stdout and progress on stderr.
- A job follower that names the stage in progress for prep, training, and evaluation runs.
- Config resolution with a written precedence order.
- Three small additive backend routes the CLI cannot be built without (listed and justified under
  Interfaces).

Out:

- **Any change to a service, runner, predictor, or job model.** The CLI drives the existing API. If
  a command needs behavior the API does not have, that is a different spec.
- **A second execution engine.** The CLI never runs training, evaluation, inference, or prep in its
  own process. See "In-process vs. HTTP" — this is the load-bearing decision of the phase.
- **A standalone thin-client distribution.** The CLI ships inside `orinth-backend`. Splitting it out
  needs a second Python distribution and a shared schema package; deferred, with the reason recorded
  below.
- **Auth.** `docs/ai/rules.md` still says "add auth before multi-user or network-exposed
  deployment"; the CLI inherits that, and `--backend` pointing at a non-loopback host is documented
  as unauthenticated.
- **Shell completions**, a TUI, watch mode, and any interactive prompt beyond a single
  destructive-action confirmation.
- **Frontend changes.** No route, component, token, or style is touched.

## The decisions

### In-process vs. HTTP

**The CLI is an HTTP client of the FastAPI backend. It never imports `app.services`, `app.ml`,
`app.training`, `app.container`, or `app.core.database`.** There is no `--remote` flag, because
there is no local mode to contrast it with: remote is just a different `--backend` URL.

The obvious alternative — import the container and call `DatasetService` / `TrainingService`
directly — is attractive because it needs no running server. It is wrong here for four reasons, in
descending order of how much they hurt.

1. **Two owners of the same mutable state.** The backend owns the five `ThreadPoolExecutor`s in
   `backend/app/container.py`, the subprocess registry that `training_service.cancel` signals, and
   the reconciliation that runs on startup. `main.py`'s lifespan calls
   `training_service.reconcile_stale_jobs()`, `evaluation_service.reconcile_stale_jobs()`, and
   `inference_service.reconcile_stale_jobs()`, and phase 3 states the rule those implement: *queued
   and running jobs are marked failed at startup because background subprocesses do not survive a
   restart*. A CLI process that ran a training job in-process would have its job row flipped to
   `failed` by the next `make dev` while its subprocess kept burning GPU. Nothing about that is
   fixable in the CLI.

2. **SQLite has one writer.** `backend/app/core/database.py` builds the engine with
   `check_same_thread: False` and no journal-mode pragma, so the default rollback journal takes an
   exclusive database-wide lock for the duration of a write, with sqlite3's default 5-second busy
   timeout. One process is the design. A CLI writing job rows while uvicorn serves the studio is a
   `database is locked` waiting to happen, and the failure would land on whichever of the two the
   user was watching.

3. **`docs/ai/rules.md`: "Avoid importing TensorFlow, Ultralytics, OpenCV, or PyTorch at module
   import."** An HTTP client satisfies that rule by construction — it imports `argparse`, `json`,
   `tomllib`, and `httpx`, and nothing else. The in-process design satisfies the letter of the rule
   and violates its point. `app.container` transitively imports `numpy` and `PIL` at module scope
   (`services/metrics.py`, `services/evaluation/helpers.py`, `services/datasets/items.py`,
   `services/datasets/preprocess.py`), `app.core.database` builds a SQLAlchemy engine *and* mkdirs
   the database directory as an import side effect, and the predictors' lazy imports only defer TF
   and Torch until the first prediction — which is exactly what `orinth infer` does on line one.
   `orinth --help` has no business paying for any of it.

4. **Ctrl-C should not kill an hour of training.** Over HTTP the job belongs to the server, so
   interrupting the follower detaches from a run that keeps going — the same semantics as closing
   the browser tab, which is the behavior every user of this platform already has. In-process,
   Ctrl-C would destroy the run, and the only way back would be to reimplement the supervision the
   server already has.

What this costs, stated honestly:

- **A server must be running.** `orinth serve` exists precisely so that is one command rather than a
  uvicorn incantation, and every command that cannot reach a backend exits **4** with the URL it
  tried, where that URL came from, and how to start one. `--wait-backend SECONDS` polls `/health` so
  a CI script can do `orinth serve & orinth doctor --wait-backend 60`.
- **Ingest re-uploads bytes that are already on disk.** A 5 GB folder goes through a loopback socket
  instead of a rename. Phase 21 measured where prep time actually goes — 20.2 s of a 21.7 s run in
  `apply`, dominated by ~45,000 file creates and ~45,000 renames — so a loopback copy is very
  unlikely to be the bottleneck, but this is a measurement to take on the first large ingest, not a
  claim. If it proves wrong, the fix is a `--link` mode on the ingest endpoint, not a local
  execution engine.

### Command dispatch stays lazy

Being an HTTP client is not by itself enough: an `argparse` tree that builds every subparser at
startup imports every command module, and one of them (`serve`) imports uvicorn and the FastAPI app.
So dispatch is two-phase:

- `app/cli/main.py` holds a **static table** of group name → module path *string* → one-line
  description. `orinth --help`, `orinth version`, and an unknown group are answered from that table
  with no import at all.
- Phase 1 parses global flags and the group token from `sys.argv` with a small parser and
  `parse_known_args`.
- Phase 2 `importlib.import_module`s **only that group's module**, calls its `build_parser()`, and
  parses the remaining arguments.
- `app/cli/client.py` imports `httpx`; it is imported by command modules, never by `main.py`.
- `app/cli/commands/serve.py` imports `uvicorn` and `app.main` **inside its handler function**. That
  is the single sanctioned heavy import in the package, and it is the one command whose entire job
  is to start the server.

This is enforced, not merely intended: `backend/tests/test_cli_dispatch.py` asserts that after
`main(["--help"])`, `main(["dataset", "--help"])`, and a stubbed `main(["dataset", "ls"])`,
`sys.modules` contains none of `tensorflow`, `torch`, `ultralytics`, `cv2`, `numpy`, `PIL`,
`sqlalchemy`, `fastapi`, `uvicorn`, `app.container`, or any `app.services.*` module. That test is the
reason this design stays true after the tenth command is added.

### Packaging and entrypoint

```toml
# backend/pyproject.toml
[project.scripts]
orinth = "app.cli.main:main"
```

`[tool.hatch.build.targets.wheel]` already declares `packages = ["app"]`, so `app/cli/` ships with
the wheel and no build configuration changes.

Invocation:

- **In the repo:** `cd backend && uv run orinth <command>`. `uv run` puts the project's console
  scripts on `PATH` inside the managed venv, so no `python -m` and no `PYTHONPATH`.
- **From an activated venv** (`source backend/.venv/bin/activate`, which the README's Backend
  section already tells users to do): plain `orinth <command>`.
- **Globally:** `uv tool install ./backend` works and is documented, with the caveat that it installs
  the full dependency set — TensorFlow, Torch, Ultralytics, transformers — for a client that uses
  `httpx`. That is the price of one distribution, and it is the reason a thin `orinth-cli` wheel is
  in Deferred rather than dismissed. Nothing in this design blocks it: the CLI imports nothing from
  the backend today, so extracting it later is a packaging change, not a rewrite.
- **No `make` target.** `make` is the repo's dev-loop entry (`make dev`, `make check`); the CLI is a
  user-facing tool, and wrapping `uv run orinth` in `make orinth ARGS=...` is worse than typing it.

**Why `backend/app/cli/`:** `docs/ai/workflow.md` says backend changes go under `backend/app`, and
the package layout there is one directory per concern — `api/` (server side), `services/` (domain
logic), `ml/`, `training/`, `core/`. The CLI is a *client*, so it is a sibling of `api/`, not a child
of it: putting it under `api/` would put a thing that makes HTTP requests next to the things that
answer them. `backend/scripts/export_openapi.py` is not a counter-precedent — it is run by path
(`python ../backend/scripts/export_openapi.py`) and never installed, so it cannot back a
`[project.scripts]` entry, which needs an importable module inside the packaged `app`.

### Argument parser: `argparse`

**stdlib `argparse`. No new dependency.**

- `backend/pyproject.toml`'s dependency list already resolves to roughly 2.7 GB installed (phase 18
  measured the desktop bootstrap at that size). Adding `typer` — which requires `click`, and whose
  useful form pulls `rich` and `shellingham` — to save some boilerplate in a package whose whole job
  is boilerplate is the wrong trade.
- Import cost is the deciding factor, not taste. `argparse` is a couple of milliseconds. `click` is
  an order of magnitude more, and `rich` an order beyond that — a large fixed cost on `orinth
  dataset ls`, paid to render a table this spec then has to constrain anyway.
- `rich` actively fights the output contract below. It wants to own the terminal and detect its
  width; here stdout must stay pipe-clean and machine-readable while stderr carries the animated
  part. Writing ~120 lines of column padding in `app/cli/output.py` is less work than configuring a
  renderer not to render.
- `argparse` subparsers map exactly onto the two-level `orinth <group> <verb>` surface, and
  `parse_known_args` is what makes the lazy two-phase dispatch above possible without a custom
  parser.

The known cost: `argparse`'s errors are terse and it hard-exits rather than raising. Both are
accepted rather than worked around — its default exit status for a parse error is **2**, which is
exactly the usage-error code this spec wants, so the CLI adopts it instead of overriding it. Where
a message would be unhelpfully bare (an unknown `--model` option id, an unknown task), the command
catches it before argparse does and prints the valid values.

### Config and workspace resolution

Precedence, highest wins, **first hit only — no merging across tiers**:

1. An explicit flag: `--backend URL`, `--project ID`, `--workspace DIR`.
2. Environment: `ORINTH_BACKEND`, `ORINTH_PROJECT`, `ORINTH_WORKSPACE`.
3. The nearest `.orinth.toml`, searching from `--workspace` (or the current directory) upward to
   `$HOME` or the filesystem root, whichever comes first.
4. The user config: `~/.config/orinth/config.toml`, and on macOS additionally
   `~/Library/Application Support/orinth.ai.studio/cli.toml` (see the desktop edge case).
5. Built-in defaults: backend `http://127.0.0.1:8000`, project `default-research-project`
   (`DEFAULT_PROJECT_ID`).

`tomllib` is stdlib on Python 3.11, so the config file costs no dependency. Every command prints its
resolved backend and project on stderr under `-v`, and `orinth doctor` always prints them *with the
tier they came from* — "resolved from ORINTH_BACKEND", "resolved from /Users/x/proj/.orinth.toml" —
because a CLI that talks to the wrong server silently is the worst failure mode this design has.

**`--workspace` names a project directory, not a storage root.** This is worth being blunt about:
"workspace" in this repo could plausibly mean `STORAGE_DIR`, and it deliberately does not. The
storage root, the models directory, the datasets directory, and the database are the *server's*
configuration, read by `Settings` from `.env` exactly as they are today. The CLI never resolves them,
never reads them, and never writes into them — that would be the second-owner problem from decision
1 arriving through the back door. `--workspace` only tells the config search where to start looking
for a `.orinth.toml`. `orinth doctor` reports the server's storage root by *asking the server*.

The one exception is `orinth serve`, which starts the backend and therefore does configure it:
`--storage-dir`, `--models-dir`, `--datasets-dir`, and `--database-url` set the corresponding
environment variables before importing `app.main`, which is how phase 18's desktop supervisor already
relocates the data root.

`.orinth.toml`:

```toml
backend = "http://127.0.0.1:8000"
project = "default-research-project"

[defaults]
json = false
```

**Project resolution has one convenience and one refusal.** When no project is configured at any
tier and the backend reports exactly one project, the CLI uses it and says so on stderr. When it
reports more than one, the command exits **2** and lists the ids — guessing which of several
workspaces a `train` command belongs to is not a guess worth making.

### Output contract

- **stdout is data. stderr is everything else.** Progress lines, stage names, warnings, resolution
  notes, and errors go to stderr, so `orinth dataset ls --json | jq '.[].id'` works with the
  progress still visible in the terminal, and `orinth infer m img.png > results.txt` leaves the
  per-file progress on screen.
- **Human by default.** Aligned columns, an uppercase header row, values truncated with `…` to the
  terminal width.
- **Off a TTY, tab-separated and untruncated**, header row retained (suppress with `--no-header`), so
  `cut -f2` and `awk -F'\t'` work on piped output. Padding is for eyes; tabs are for `awk`.
- **`--json` on every command that returns data.** The emitted document is the API response body,
  pretty-printed, verbatim — the CLI adds no fields, renames none, and reorders none. Commands that
  aggregate several calls emit a single object whose values are verbatim schemas
  (`{"dataset": …, "readiness": …, "plan": …}`). This is what keeps `--json`, `frontend/types/api.ts`,
  and `backend/app/schemas.py` in agreement with no third schema to maintain.
- **`--jsonl` where a stream is the natural shape** (`infer` over a directory, `test --per-item`):
  one JSON document per line, so a 10,000-file batch can be consumed without buffering. Only where
  it earns itself; `--json` remains available on the same commands.
- **The CLI does not validate response bodies.** No `pydantic` import, no `app.schemas` import. It
  reads the keys it needs and passes the rest through, so a CLI talking to a newer backend renders
  what it understands and still emits the complete body under `--json`. Importing `app.schemas` would
  be cheap and would couple the CLI to one build; that trade is refused.
- **Color** only when stdout is a TTY, `NO_COLOR` is unset, and `--json`/`--jsonl` is not in effect.
  `--no-color` forces it off. Color is used for one thing: readiness and job status words.
- **`| head` must not traceback.** `BrokenPipeError` on stdout is caught and exits 0 quietly.

### Exit codes

| Code | Meaning | Produced by |
| --- | --- | --- |
| `0` | Success | Everything that completed |
| `1` | Runtime failure | Server returned 4xx/5xx, a job finished `failed`, a local IO error, any input in a batch failed |
| `2` | Usage error | argparse's own parse failure, an unknown `--model` option id, an ambiguous project, a malformed `.orinth.toml` |
| `3` | **Not ready to train** | `readiness.trainable == false` — from `dataset readiness`, from `dataset prep` finishing in `needs_input`, and from `train`'s preflight |
| `4` | Backend unreachable | Connection refused, DNS failure, or `--wait-backend` expiring |
| `130` | Interrupted | SIGINT during a follow. The server-side job keeps running |

`3` is separate from `1` for one concrete reason: a CI pipeline treats them differently. "Your data
is not trainable" is a gate the author can fix; "the server fell over" is an infrastructure retry.
Collapsing them makes `orinth dataset readiness $ID || retry` wrong. `4` exists for the same reason —
`orinth serve &` racing a first command is a retry, not a failure.

### Progress reporting for long jobs

Prep, training, and evaluation all run on the backend's per-domain thread pools and are polled by the
web UI. The CLI polls the same endpoints. It does not stream: there is no SSE or websocket for job
progress today, and inventing one for the CLI would be a server change this phase's Scope excludes.

What it polls, and how often:

- **Prep** → `GET /api/datasets/{id}/prep/status`. Phase 21 chose this endpoint over `GET /datasets`
  for exactly this purpose and measured it at 4 ms against a 2–3 s catalog build, so the CLI must use
  it and must never poll the catalog in a loop.
- **Training** → `GET /api/training/jobs/{id}`. **Evaluation** → `GET /api/testing/jobs/{id}`.
- Cadence: every 1 s, backing off to 5 s once neither `percent` nor `current_step` has changed for 60
  s. Capped at 5 s so a finished job is noticed promptly.

What it prints, on stderr:

- **One line per stage change, not per poll.** The backend already names the stage — `JobProgress`
  carries `current_step` and `current_item`; `DatasetPrepStatus` carries `step` ("staging",
  "detecting", "planning", "transforming", "applying", "splitting") and a `detail` sentence written
  for a waiting human ("Reading 312 files"). A spinner would throw that away.

  ```
  12:04:31  planning    Reading 312 files
  12:04:33  applying    Writing items  4200/15000   28%   eta 42s
  12:05:15  splitting   Distributing 15000 items into train/valid/test
  ```

- **A keepalive every 30 s** when nothing has changed, carrying elapsed time, so a long silent stage
  (a multi-GB base-model download, a 20-second `apply`) is distinguishable from a hang.
- **On a TTY** the in-progress line is redrawn in place with `\r`; **off a TTY** every change is a new
  line and `\r` is never emitted, so CI logs stay diffable.
- **New log lines only.** `JobProgress.logs` is a growing list; the follower tracks how many it has
  printed and prints only the suffix. Printed under `-v`, and always on failure — the last 20 lines,
  which is what makes a failed run diagnosable without a second command.
- **Training metric digest.** After each new `history` row, one line built from whatever numeric
  columns that row has: `epoch 7/50  loss 0.412  val_loss 0.501  mAP50 0.63`. Column names differ by
  family, so the CLI selects by preference list and falls through — the same principle
  `TrainingChart` uses to group series, and the reason no family is hardcoded here.

`--follow` is the **default** for `prep`, `train`, and `test`. `--detach` prints the job id (or
dataset id) on stdout and exits 0 immediately, which is the CI-friendly form when a later step will
collect the result. `orinth train logs <id> --follow` and `orinth test logs <id> --follow` reattach.

**Ctrl-C detaches; it does not cancel.** A SIGINT handler prints the reattach command and exits 130,
leaving the job running server-side. A second SIGINT exits immediately. Cancelling is
`orinth train cancel <id>`, an explicit act, because a reflex to stop watching must not destroy an
hour of GPU time. `--cancel-on-interrupt` opts into the other behavior for a user who wants it.

## Interfaces

### Module layout

```
backend/app/cli/
  __init__.py       version string only; imports nothing
  main.py           entrypoint; global flags, the static group table, two-phase dispatch, exit codes
  config.py         precedence resolution (flag → env → .orinth.toml → user config → default); tomllib
  client.py         httpx wrapper: base URL, timeouts, HTTP error → CliError, multipart batching, streaming download
  output.py         table renderer, --json/--jsonl emitters, TTY + color detection, the stdout/stderr split
  progress.py       the job follower: poll cadence, stage-change printing, log-tail tracking, SIGINT handling
  errors.py         CliError and the exit-code table
  commands/
    __init__.py     empty; group modules are imported by path string, never re-exported
    dataset.py      ls show ingest prep readiness export rm
    train.py        run (default verb) ls show logs cancel rm models
    test.py         run (default verb) ls show logs compare per-item rm   [alias: eval]
    infer.py        run (default verb) ls show rm
    model.py        ls show download rm
    project.py      ls show create
    serve.py        the only module importing uvicorn / app.main — inside its handler
    doctor.py       runtime diagnosis
```

### API endpoints

Everything the CLI does reuses routes the UI already calls, with **three exceptions**. Each is
additive, breaks nothing, and is useful to the UI too — none is a CLI-only backdoor.

1. **`GET /api/datasets/{dataset_id}` → `DatasetSummary`.** There is no single-dataset route today;
   `frontend/lib/api/datasets.ts` gets one dataset by fetching the whole catalog and filtering. That
   is fine for a page that wants the catalog anyway and wrong for `orinth dataset show` and for every
   post-job readiness re-read, because phase 21 measured `GET /datasets` at 2–3 s on a large upload
   and 4.5 s on the live storage tree — `_split_summary` walks items and reads annotation JSON per
   item, for every dataset. The implementation is two lines returning
   `dataset_service.summary(dataset_id)`, which `GET /{dataset_id}/readiness` already calls.
   **Declaration order matters:** it must be registered *after* `/hub/search`, `/hub/preview`,
   `/import/hub`, and `/ingest`, or the path parameter swallows those literals — the same trap the
   router's existing comments call out.

2. **`GET /api/datasets/{dataset_id}/download` → zip.** `orinth dataset export` cannot be built from
   existing routes: `POST /{id}/versions` writes a snapshot to a server-side path
   (`DatasetVersionSummary.path`) and nothing streams it back. This mirrors
   `GET /api/models/{model_id}/download`, which already bundles multi-asset models as a zip under
   ignored storage, and it gives the dataset catalog the download action the model catalog has. Query
   parameters: `splits` (comma-separated, default all) and `version_id` (export a snapshot instead of
   the live root).

3. **`GET /health` gains `version` and `app`.** It returns `{"status": "ok"}` today. `orinth version`
   exists to tell a user whether the CLI and the server it is talking to are the same build, which it
   cannot do without this. The change is additive; phase 18's desktop supervisor polls `/health` for a
   200 and ignores the body, so nothing breaks. Return type stays `dict[str, str]`.

Everything else is existing surface:

| CLI | Route |
| --- | --- |
| `dataset ls` | `GET /api/datasets?project_id=` |
| `dataset ingest` | `POST /api/datasets/ingest`, `POST /api/datasets/{id}/ingest` |
| `dataset prep` | `POST /api/datasets/{id}/prep`, `GET /api/datasets/{id}/prep/status`, `GET /api/datasets/{id}/prep` |
| `dataset readiness` | `GET /api/datasets/{id}/readiness` |
| `dataset rm` | `DELETE /api/datasets/{id}` |
| `train models` | `GET /api/training/model-options?task_type=` |
| `train` | `POST /api/training/jobs`, `GET /api/training/jobs/{id}` |
| `train cancel` / `rm` | `POST /api/training/jobs/{id}/cancel`, `DELETE /api/training/jobs/{id}` |
| `test` | `GET /api/testing/datasets`, `POST /api/testing/jobs`, `POST /api/testing/jobs/batch` |
| `test compare` / `per-item` | `GET /api/testing/jobs/{id}/comparison`, `GET /api/testing/jobs/{id}/per-image` |
| `infer` | `POST /api/inference`, `GET /api/inference` |
| `model` | `GET /api/models`, `GET /api/models/{id}/download`, `DELETE /api/models/{id}` |
| `project` | `GET /api/projects`, `POST /api/projects` |
| `doctor` | `GET /health`, `GET /api/projects`, `GET /api/models`, `GET /api/training/llm/environment` |

### Schemas

**None added.** The CLI reads JSON dictionaries and never constructs a Pydantic model. The three
routes above reuse `DatasetSummary`, `FileResponse`, and a plain dict.

### Storage / DB

**None.** The CLI writes exactly two kinds of file, both only where told: an export archive at
`--to`, and inference overlays at `--out`. It never writes under `storage/`.

## Command surface

Global flags, accepted by every command: `--backend URL`, `--project ID`, `--workspace DIR`,
`--json`, `--jsonl` (where supported), `--no-header`, `--no-color`, `-q/--quiet`, `-v/--verbose`,
`--timeout SECONDS` (HTTP read timeout, default 60 — job following is not bounded by it),
`--wait-backend SECONDS`.

### `orinth dataset`

- **`ls [--task T] [--state ready|needs_prep|needs_input|blocked] [--limit N] [--json]`**
  Columns: `ID  NAME  TASK  FORMAT  TRAIN  VALID  TEST  READY`. `READY` is `readiness.state`.
  Exit 0 always, including on an empty list.

- **`show <id> [--json]`**
  Identity, labels, split counts, the readiness check list, and the prep state and plan when one
  exists. `--json` emits `{"dataset": …, "plan": …}`, the plan omitted when there has been no run.
  Informational: always exit 0 when the dataset exists.

- **`ingest <path>... [--name NAME] [--into DATASET_ID] [--prep] [--batch N] [--all] [--json]`**
  - A `<path>` may be a **file**, a **directory**, or an **archive**.
  - A directory is walked recursively and each file's path relative to that directory becomes its
    `relative_paths[i]` entry. That list is not cosmetic: phase 21 established that the directory
    layout *is* the detection signal — `data.yaml` beside `train/labels/` means YOLO, sibling
    `normal/` and `kista/` folders mean classification with those labels — and an upload without it
    degrades to a flat pile of files.
  - **Archives (`.zip`, `.tar.gz`, `.tgz`, `.tar`) are expanded client-side** into a temp directory
    and uploaded as a folder. Phase 21 deferred `.zip` ingest server-side for want of an extraction
    guard; doing it in the CLI gets the feature without adding an untrusted-extraction path to the
    server, because the archive is the user's own file on the user's own machine. The guard still
    applies: members with absolute paths, `..` segments, or symlinks are refused, and expansion is
    capped at 20,000 members and 5 GB uncompressed (`--max-files`, `--max-bytes`).
  - `.git/`, `__MACOSX/`, `.DS_Store`, and dotfiles are skipped unless `--all`.
  - Files are uploaded in batches (`--batch`, default 200 per request) so a 15,000-file folder is not
    one multipart request, with `staged 1200/3184` on stderr.
  - stdout gets the dataset id, alone, so `DS=$(orinth dataset ingest ./data)` works. `--json` emits
    the `DatasetSummary`.
  - `--prep` chains straight into `prep` and follows it, which is the one-command form of the whole
    flow.
  - Every file rejected → server 422 → exit 1 with its message.

- **`prep <id> [--no-apply] [--detach] [--wait] [--json]`**
  `POST /prep` answers 202 with the dataset already reading `planning`; the CLI then polls
  `/prep/status` and prints the stage lines above. On a terminal `ready`, stderr gets the plan summary
  — task, format, labels, split counts, the engine that produced it, and any `notice` — and stdout
  gets the `DatasetSummary` (or `{"dataset": …, "plan": …}` under `--json`).
  - `--no-apply` sends `auto_apply: false`, stopping at `planned` for review.
  - A 409 (`PrepBusyError` — a run is already in flight) exits 1 with the server's message;
    `--wait` attaches to the in-flight run instead of failing.
  - **Exit 3 when the run finishes and the dataset is still not trainable**, with the plan's
    `needs_input` reason on stderr. Phase 21's flat-unlabelled-images case lands here, and it is the
    outcome CI must be able to see.

- **`readiness <id> [--advisory-fatal] [--json]`**
  Prints the check table: `RESULT  SEVERITY  ID  LABEL  DETAIL`, then `readiness.summary`.
  **Exit 0 when `trainable`, 3 otherwise.** `--advisory-fatal` also fails on advisory checks (the
  undrained inbox, class imbalance), for a pipeline that wants them to be errors. `busy: true` is
  reported but does not change the exit code — phase 21 is explicit that a non-destructive
  detect/plan in flight leaves the dataset exactly as trainable as it was. `state == "blocked"`
  exits 3.
  `--json` emits the `DatasetReadiness` verbatim.

- **`export <id> --to PATH [--splits train,valid,test] [--version ID]`**
  Streams `GET /{id}/download` to `PATH`, or to stdout when `PATH` is `-` — the single case where
  stdout carries bytes rather than a table, and it is opt-in. Prints bytes written on stderr.

- **`rm <id>... [--yes] [--json]`**
  `DELETE /datasets/{id}` per id. **Refuses without `--yes` when stdin is not a TTY**, so a CI typo
  cannot delete a dataset; prompts once when it is. Exit 1 if any deletion failed, and the report
  says which succeeded.

### `orinth train`

**`orinth train <dataset-id> --task T --model OPTION_ID [--epochs N] [--batch-size N]
[--image-size N] [--lr F] [--device D] [--name NAME] [--set k=v]... [--force] [--detach] [--json]`**

- **Preflight is the headline feature.** Before creating anything, the CLI reads
  `GET /datasets/{id}` and, if `readiness.trainable` is false, **exits 3** printing
  `readiness.summary` and the failing checks. Phase 21's motivating bug is the reason:
  `keras_common.load_split` silently skips images with no annotation file, so an unlabelled
  classification dataset trains on zero items and reports success. The CLI refuses to start that run.
  `--force` skips the preflight and says on stderr exactly which checks it skipped.
- `--model` takes a `model_option_id`. An unknown one exits **2** and prints the valid ids for the
  chosen task, fetched from `GET /training/model-options?task_type=`. `orinth train models --task T`
  lists them with their runnable/gated state.
- `--set k=v` writes into `hyperparameters` (repeatable), each value parsed with `json.loads` and
  falling back to the raw string. That reaches every phase-13 and phase-14 advanced parameter —
  `--set lora_r=32 --set finetune_method=qlora` — without a flag per knob and without this spec
  enumerating a catalog that changes.
- `POST /training/jobs`, then follow `GET /training/jobs/{id}`.
- On completion stdout carries the terminal `TrainingJobRead` under `--json`, or a summary block:
  job id, status, `promoted_model_id`, a final metrics table, and the artifacts directory.
- Exit 0 on `completed`, 1 on `failed` (with `error` and the last 20 log lines on stderr), 130 on
  interrupt.
- `train ls | show <id> | logs <id> [--follow] | cancel <id> | rm <id>`.

**Why no shared `orinth job` group.** Training, evaluation, and inference jobs live in three tables
behind three route prefixes. A single `orinth job show <id>` would have to try all three and guess,
and would report the wrong "not found" for a typo. The verbs live under the group that owns the job.

### `orinth test` (alias `orinth eval`)

**`orinth test <model-id>... --dataset <dataset-id> [--split test] [--limit N]
[--fail-under metric=value]... [--per-item] [--detach] [--json]`**

- The dataset id and split are resolved to an evaluation `dataset_key` through
  `GET /testing/datasets` — **never** by string-building `dataset:<id>:<split>`. The reference splits
  are keyed `yolo_test` and `coco_test` and do not follow that shape, and a hand-built key for an
  incompatible split produces a 404 the user cannot act on. Phase 2 also restricts editable datasets
  to their `test` split, and that restriction is expressed in this endpoint's output, so honouring it
  is free.
- One model → `POST /testing/jobs`. Several → `POST /testing/jobs/batch`, which shares a
  `comparison_id`. The follower polls every job and prints one progress line per job.
- **Metrics reported**, task-aware, from `EvaluationJobRead.metrics` — the CLI prints the keys the
  server actually produced, ordered by a per-task preference list, and everything else under `-v`. It
  invents no metric and computes none.
  - classification / text_classification: accuracy, balanced accuracy, macro F1, weighted F1, MCC,
    Cohen κ, macro AUC when scores were available
  - object_detection / segmentation: pixel Dice, IoU, precision, recall, specificity; object
    precision/recall/F1 at the IoU 0.2 threshold `docs/ai/rules.md` fixes
  - summarization and question_answering: whatever the evaluation service reports for them
- Several models print one row per model over the shared column set from
  `GET /jobs/{id}/comparison`.
- `--per-item` prints `GET /jobs/{id}/per-image` rows; beyond 50 rows it requires `--json`/`--jsonl`,
  because a thousand-row table in a terminal is not output, it is scrollback.
- **`--fail-under metric=value`** (repeatable) exits 1 when a reported metric is below the threshold.
  That is the CI gate for models, the counterpart to `readiness`'s gate for data. A metric name the
  job did not report exits 2, not 0 — silently passing a gate that never ran is the failure mode
  worth designing against.
- Exit 1 if any job failed or any threshold was missed; 0 otherwise.

### `orinth infer`

**`orinth infer <model-id> <input>... [--text STRING] [--question Q] [--confidence F] [--iou F]
[--max-length N] [--recursive] [--jobs N] [--out DIR] [--json|--jsonl]`**

- `<input>` is a file path, a directory (batch; `--recursive` to descend), or `-` to read text from
  stdin. `--text` supplies text inline for NLP models.
- Each input is one `POST /api/inference`, the **synchronous** route. Single-image prediction is
  seconds, so the job route's polling round-trip buys nothing; `POST /inference/jobs` is what the
  studio uses because it wants a progress bar in a browser. Batch is the same route once per file,
  `--jobs N` parallel requests, default **1**. This spec does not add a batch endpoint: the work is
  server-side model inference, one predictor instance deep, and N parallel HTTP requests cannot make
  it faster than the server is. The useful ceiling for `--jobs` is a measurement to take, not a
  number to assert; it is capped at 8 so a mistake cannot flood the server.
- Output, one row per input on stdout: `INPUT  LABEL  SCORE  DETECTIONS  MS`. For text tasks `LABEL`
  is the `nlp_result` head (predicted class, answer, or the first clause of a summary), truncated.
  `--json` emits an array of verbatim `InferenceResult` objects; `--jsonl` one per line, which is what
  a large batch wants.
- `--out DIR` downloads each result's `overlay_url` into `DIR`, named after the input file. Image
  tasks only; served by the existing `/media` mount.
- Exit 0 when every input succeeded, 1 if any failed. stderr closes with `47 ok, 3 failed`.

### `orinth model`

`ls [--task T] [--available] [--json]`, `show <id>`, `download <id> --to PATH`, `rm <id> [--yes]`.
Thin, and it earns its place for one reason: `train` and `infer` both take a model id, and without
`model ls` there is no way to find one without opening a browser.

### `orinth project`

`ls`, `show <id>`, `create --name N [--task T]...`. Equally thin, and needed because `--project`
takes an id, the ambiguity refusal above requires a way to list them, and a CI bootstrap needs a way
to create one.

### `orinth serve`

**`orinth serve [--host H] [--port P] [--reload] [--storage-dir DIR] [--models-dir DIR]
[--datasets-dir DIR] [--database-url URL]`**

Starts the FastAPI app in the foreground. **It earns its place** because the HTTP-first decision
makes "no backend running" the CLI's most common failure, and this turns the fix into one command
instead of `cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`. It is the
only place `app.main` and `uvicorn` are imported, inside the handler.

Foreground only — no `--detach`, no PID file, no supervision. A daemon is a lifecycle to own, and
`orinth serve &` plus `orinth doctor --wait-backend 60` is what CI needs and all it needs. Phase 18
already owns real supervision and does it in Rust for reasons this command should not relitigate.

### `orinth doctor`

Reports, and this is deliberately different from `make doctor`, which checks *build* tooling (uv,
pnpm, Python 3.11, Node): `orinth doctor` checks the **runtime**.

```
backend      OK    http://127.0.0.1:8000  (from ORINTH_BACKEND)  version 0.1.0
project      OK    default-research-project  "Research workspace"
datasets     12 total, 8 ready to train, 1 needs input
models       23 registered, 9 available
accelerator  mps   (peft backend; 4-bit QLoRA unavailable on MPS)
storage      /Users/x/orinth/storage
```

Accelerator comes from `GET /api/training/llm/environment` — the existing probe, whose torch import
is already lazy inside its handler. Exit 0 when the backend is reachable, 4 when not. It earns its
place as the one command that answers "why isn't this working", and specifically as the one place the
CLI prints *where each resolved setting came from*.

### `orinth version`

CLI version from `importlib.metadata.version("orinth-backend")`, plus the backend's from `/health`
when reachable. Prints both and warns on stderr when they differ, which is the only reason the
command exists. Works with no server. `--version` is a synonym.

## Data Flow

```
argv
 │
 ├─ phase 1: global flags + group token           (argparse, no imports)
 │
 ├─ config.resolve()  flag → env → .orinth.toml → user config → default
 │
 ├─ phase 2: importlib.import_module("app.cli.commands.<group>")
 │              build_parser() → parse verb + flags
 │
 ▼
client.request()  ──HTTP──►  FastAPI  ──►  app.services.*  ──►  storage/ + app.db
 │                                              (the server's process, its executors,
 │                                               its subprocesses, its SQLite writer)
 │
 ├─ short call ──► output.table() / output.json()   ──► stdout
 │
 └─ job created ──► progress.follow(poll_url)
                      │  every 1s → 5s, until a terminal status
                      ├─ stage change / keepalive / new log lines ──► stderr
                      └─ terminal object                          ──► stdout
                                                                       │
                                                        exit 0 | 1 | 3 | 4 | 130
```

## Edge Cases

- **No backend running.** Exit 4. The message names the URL tried, the tier it was resolved from, and
  two ways forward (`orinth serve`, `make dev`). No implicit retry; `--wait-backend S` polls
  `/health` for S seconds and is what a CI script uses right after `orinth serve &`.
- **A dataset id that does not exist.** 404 → exit 1. On the error path only, the CLI makes one
  `GET /datasets` call and suggests the closest ids by substring match. One extra request on a path
  that has already failed is worth a usable message.
- **A job fails mid-run.** The follower stops at the terminal status, prints `error` plus the last 20
  `progress.logs` lines on stderr, exits 1. `--json` still emits the terminal job object, so CI can
  archive the failure rather than re-query it.
- **Ctrl-C during a long job.** The job keeps running server-side — it is the server's thread or
  subprocess, not the CLI's. stderr prints the reattach command; exit 130. A second Ctrl-C exits at
  once. Cancelling requires `orinth train cancel <id>`.
- **The backend restarts while a job is followed.** The next poll gets a connection error; the
  follower retries for 30 s before exiting 4. If the server comes back, phase 3's rule applies and the
  job will read `failed` on the next poll — the CLI reports that as the job's own outcome (exit 1),
  not as a CLI error, and says the run did not survive the restart.
- **A second prep run on the same dataset.** Server 409 (`PrepBusyError`); exit 1, or attach with
  `--wait`.
- **A task the project has not declared.** `POST /prep/apply` returns 409; exit 1 with the server's
  message, which already names project settings.
- **A machine with no GPU.** Nothing special. `doctor` reports the probed device, and `train` warns
  once on stderr when `--device cuda` disagrees with the probe. The warning never blocks: the server
  is the authority on its own hardware and the probe is advisory.
- **The Tauri desktop build, where `orinth` is not on `PATH`.** Two separate problems, both answered.
  - *Not on PATH.* The `.dmg` provisions a venv at
    `~/Library/Application Support/orinth.ai.studio/runtime/backend/.venv`, so the console script
    exists at `<that>/bin/orinth`. **The desktop app does not install a global `orinth`.** Writing to
    `/usr/local/bin` needs admin rights, and phase 18 ships an ad-hoc-signed, un-notarized bundle
    that must not be asking for them. `desktop/README.md` documents the one-line `ln -s` a user can
    run themselves.
  - *The wrong default backend.* The desktop supervisor binds **ephemeral** loopback ports, so the
    CLI's `http://127.0.0.1:8000` default would reach either nothing or some other project's server —
    exactly the failure phase 18 hit with the baked proxy origin. The app therefore writes its chosen
    backend port to `~/Library/Application Support/orinth.ai.studio/cli.toml` on each launch, in the
    `.orinth.toml` shape, and the config search reads that path at the user-config tier on macOS.
    This closes the loop without the app touching `PATH` or the CLI guessing a port.
- **`orinth dataset ls | head -5`.** `BrokenPipeError` is caught; exit 0, no traceback.
- **stdout is not a TTY.** Tab-separated, untruncated, no `\r`, no color.
- **A newer backend returning unknown fields.** The CLI reads the keys it needs and passes the body
  through under `--json`; no client-side validation, so a schema addition never breaks an older CLI.
- **An older backend missing the three new routes.** `dataset show` falls back to filtering
  `GET /datasets` (slow but correct), `version` reports the backend version as unknown, and
  `dataset export` exits 1 naming the required backend version. A 404 on a route the CLI expects
  produces "this backend is older than this CLI", not a bare 404.
- **A malformed `.orinth.toml`.** Exit 2 naming the file and the parse error. Silently falling back
  to a default would send the command to the wrong server.
- **Ambiguous project.** More than one project and none configured → exit 2 listing the ids.
- **An archive with a `..` member or a symlink.** Refused before extraction, exit 1, naming the
  member.
- **An empty ingest.** A path that resolves to zero eligible files exits 2 before contacting the
  server, saying how many were skipped and why (hidden, filtered, or empty directory).

## Acceptance Criteria

Behavior:

- `orinth --help` runs without importing TensorFlow, Torch, Ultralytics, OpenCV, numpy, PIL,
  SQLAlchemy, FastAPI, uvicorn, `app.container`, or any `app.services` module. Asserted by a test,
  not by inspection.
- `orinth dataset ingest ./folder --prep` takes an unmodified YOLO folder, a per-class image folder,
  an alpaca JSONL, or a labelled CSV from a shell to a dataset that `orinth dataset readiness` exits
  0 on — the same four inputs phase 21 accepts through the UI.
- `orinth dataset readiness <id>` exits 3 on a dataset that is not trainable and prints the failing
  checks; exits 0 when it is. It is usable as `orinth dataset readiness $DS || exit 1` in CI.
- `orinth train <ds> --task ... --model ...` refuses with exit 3 on a dataset whose readiness is not
  trainable, before creating a job row.
- Following a prep, training, or evaluation run prints the **name of the current stage** and its
  counts, updating as the stage changes, and a keepalive during a long silent stage. Not a spinner.
- Ctrl-C during a follow exits 130 and leaves the job running; `orinth train show <id>` afterwards
  reports it still running.
- Every command that returns data supports `--json`, and its output is the API response body
  verbatim — byte-comparable against a direct `curl` of the same route.
- stdout carries only data: `orinth dataset ls --json | jq -e '.[0].id'` succeeds while progress and
  warnings remain visible on the terminal.
- With no backend running, every networked command exits 4 with an actionable message; `version` and
  `doctor` still run.
- `orinth serve` starts the same app `make backend` starts, on the same default port.

Tests: see Validation.

Manual checks: the transcripts in "Worked example", run against a real `orinth serve` and the repo's
`sample_data/`.

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd backend && uv run ruff check .
cd frontend && pnpm generate:api      # three new routes change types/generated/api.ts
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```

The frontend gates run because `pnpm generate:api` regenerates `frontend/types/generated/api.ts` from
the backend's OpenAPI export, and that file is tracked. No frontend source changes.

### How the CLI is tested — and why not end to end

Every command is the same three steps: parse arguments, build a request, render a response. The
server-side behavior those requests drive is already covered — `test_dataset_prep_service.py`,
`test_dataset_readiness.py`, `test_dataset_service.py`, `test_observable_jobs.py`,
`test_training_progress.py`, the evaluation and inference suites. Re-driving training through the CLI
would import the training stack into the CLI's test run and add no coverage of the CLI.

So the CLI is tested against `httpx.MockTransport` with canned bodies shaped from
`backend/app/schemas.py`. That gives exact assertions on the wire — method, path, query, multipart
field names — and on the rendered text, in milliseconds, with no server, no SQLite, and no ML import.
One end-to-end run stays a **manual** check for the same reason `make test` does not train a model.

New tests, all under `backend/tests/`:

- `test_cli_dispatch.py` — the import guard. After `main(["--help"])`, `main(["dataset", "--help"])`,
  and a stubbed `main(["dataset", "ls", "--json"])`, `sys.modules` contains none of `tensorflow`,
  `torch`, `ultralytics`, `cv2`, `numpy`, `PIL`, `sqlalchemy`, `fastapi`, `uvicorn`, `app.container`,
  or any `app.services.*`. Also: an unknown group is answered from the static table with no import;
  `orinth dataset --help` imports exactly one command module.
- `test_cli_config.py` — precedence at every tier; a `.orinth.toml` found by walking up from a
  subdirectory; the search stopping at `$HOME`; a malformed file exiting 2 with the path; the macOS
  desktop config path read at the user tier; the ambiguous-project refusal.
- `test_cli_output.py` — column alignment and `…` truncation on a simulated TTY; tab-separated and
  untruncated off it; `--no-header`; `--json` byte-identical to the response body; color suppressed
  under `NO_COLOR`, `--no-color`, and `--json`; `BrokenPipeError` exiting 0.
- `test_cli_exit_codes.py` — one case each for 0, 1 (server 500), 2 (unknown flag; unknown
  `--model`), 3 (`readiness` on a non-trainable fixture; `train` preflight), 4 (`httpx.ConnectError`),
  130 (SIGINT during a follow).
- `test_cli_dataset.py` — `ls` filters and columns; `show` falling back to the catalog on a 404 from
  the new route; `ingest` sending `relative_paths` parallel to `files` and preserving nested paths,
  batching at `--batch`, refusing a `..` archive member, exiting 2 on an empty selection; `prep`
  polling `/prep/status` and never `GET /datasets`; `prep` exiting 3 on `needs_input`; `rm` refusing
  without `--yes` off a TTY.
- `test_cli_train.py` — the readiness preflight blocking job creation and `--force` bypassing it;
  `--set k=v` landing in `hyperparameters` with JSON typing; an unknown `--model` listing valid ids
  and exiting 2; the terminal `failed` path printing the log tail and exiting 1.
- `test_cli_test.py` — `dataset_key` resolved through `GET /testing/datasets` rather than
  string-built; several models routed to `/jobs/batch`; `--fail-under` failing below a threshold and
  exiting 2 on a metric the job never reported.
- `test_cli_infer.py` — a directory expanded to one request per file; `--jsonl` emitting one document
  per line; `--out` writing overlays named after inputs; a partial batch failure exiting 1 with the
  `ok/failed` count.
- `test_cli_progress.py` — one line per stage change and not per poll; a keepalive after 30 s of no
  change (clock injected); only new `logs` entries printed; the backoff from 1 s to 5 s; SIGINT
  detaching without cancelling.
- `test_api_routes.py` / `test_dataset_service.py` (extended) — `GET /datasets/{id}` returns the same
  object the list returns for that id and 404s on an unknown one; `GET /datasets/hub/search` still
  resolves after it is registered (the declaration-order trap); `GET /datasets/{id}/download` streams
  a zip containing the split tree; `/health` carries `version`.

### To be measured, not asserted

This is a plan. These are the numbers the implementation must produce, and none of them is claimed
here:

- `orinth --help` wall time, cold, on a machine with the full ML environment installed. The target is
  under 100 ms; the point of the design is that it should be dominated by process start.
- The per-file cost of multipart ingest to loopback against a large real folder, compared with phase
  21's measured 20.2 s `apply` on a 15,000-row table — to confirm the upload is not the bottleneck,
  or to justify a `--link` ingest mode if it is.
- Poll overhead of a follow over a long training run: requests per minute and the server-side cost of
  `GET /training/jobs/{id}`, to confirm the 1 s → 5 s backoff is right.
- A useful ceiling for `orinth infer --jobs N` against the real inference path.

## Worked example

**Intended output of a design that is not yet built.** No command below has been run; these
transcripts specify what the implementation must produce, and are the manual checks in Acceptance
Criteria.

### 1. Ingest and prepare a folder nobody has declared anything about

```console
$ cd backend && uv run orinth dataset ingest ~/Downloads/chest-xray --name "Chest X-ray" --prep
staged 400/400 files (12 directories)
dataset ds_chest-xray-4f2a91 created

14:02:11  planning     Scanning 400 staged files
14:02:12  planning     image_folder / classification — labels NORMAL, PNEUMONIA (from folder names)
14:02:12  applying     Writing items  400/400
14:02:14  splitting    Distributing 400 items into train/valid/test
14:02:15  ready        280 train / 80 valid / 40 test

Plan     classification · image_folder · 2 labels · heuristic engine (no OpenRouter key configured)
Splits   train 280 · valid 80 · test 40 · seed 42
Ready    ready to train

ds_chest-xray-4f2a91
```

### 2. Gate a pipeline on readiness

```console
$ uv run orinth dataset readiness ds_chest-xray-4f2a91
RESULT  SEVERITY   ID                LABEL                        DETAIL
pass    blocking   prep_idle         No prep run in progress
pass    blocking   prep_applied      Prep has been applied
pass    blocking   has_items         Dataset has items
pass    blocking   labels_defined    At least 2 labels
pass    blocking   train_non_empty   Train split has items
pass    blocking   train_annotated   Train items are annotated
pass    blocking   valid_non_empty   Valid split has items
pass    blocking   valid_annotated   Valid items are annotated
pass    advisory   inbox_drained     Inbox is empty

ready to train
$ echo $?
0
```

And the case CI exists to catch:

```console
$ uv run orinth dataset readiness ds_scans-loose-8801
RESULT  SEVERITY   ID                LABEL                     DETAIL
pass    blocking   prep_idle         No prep run in progress
FAIL    blocking   prep_applied      Prep has been applied     Orinth could not infer class labels
                                                               from a flat folder of images. Label
                                                               them in the Data tab.
pass    blocking   has_items         Dataset has items
FAIL    blocking   labels_defined    At least 2 labels         This task needs at least 2 class
                                                               labels; 0 defined.

not ready to train: Orinth could not infer class labels from a flat folder of images.
$ echo $?
3
```

### 3. Train, following the run

```console
$ uv run orinth train ds_chest-xray-4f2a91 --task classification \
    --model keras_efficientnetb0 --epochs 20 --name "xray-effnet-b0"
readiness  ready to train (8 blocking checks passed)
job        tj_7c1e04ab created

14:08:02  Preparing dataset      280 train / 80 valid
14:08:19  Loading base weights   efficientnetb0 imagenet
14:08:41  Training               epoch 1/20   loss 0.6612  val_loss 0.5904  accuracy 0.612
14:09:04  Training               epoch 2/20   loss 0.5218  val_loss 0.4471  accuracy 0.734
...
14:16:55  Training               epoch 20/20  loss 0.1043  val_loss 0.1876  accuracy 0.948
14:17:08  Saving                 best_model.keras
14:17:11  completed              8m 12s

Job        tj_7c1e04ab  completed
Model      md_xray-effnet-b0-19c4  (registered automatically)
Metrics    accuracy 0.948 · val_accuracy 0.921 · val_loss 0.1876 · best epoch 18
Artifacts  storage/training_runs/tj_7c1e04ab/

md_xray-effnet-b0-19c4
```

Interrupting instead:

```console
^C
Detached. The run is still going on the server.
Reattach with:  orinth train logs tj_7c1e04ab --follow
Cancel with:    orinth train cancel tj_7c1e04ab
$ echo $?
130
```

### 4. Evaluate, then infer

```console
$ uv run orinth test md_xray-effnet-b0-19c4 --dataset ds_chest-xray-4f2a91 --fail-under accuracy=0.90
dataset  dataset:ds_chest-xray-4f2a91:test  (40 items)
job      ej_2f90c7d1 created

14:19:30  Evaluating   40/40   100%

MODEL                   ACCURACY  BALANCED  MACRO F1  MCC     AUC
md_xray-effnet-b0-19c4  0.925     0.918     0.924     0.849   0.961

accuracy 0.925 >= 0.90
$ echo $?
0

$ uv run orinth infer md_xray-effnet-b0-19c4 ~/Downloads/new-scans/ --out ./overlays
INPUT             LABEL       SCORE   DETECTIONS  MS
scan-0001.png     PNEUMONIA   0.964   0           412
scan-0002.png     NORMAL      0.887   0           355
scan-0003.png     PNEUMONIA   0.712   0           349
3 ok, 0 failed · overlays written to ./overlays
```

### 5. The whole thing as a CI step

```yaml
- run: cd backend && uv run orinth serve --port 8000 &
- run: cd backend && uv run orinth doctor --wait-backend 90
- run: cd backend && uv run orinth dataset readiness "$DATASET_ID"        # exits 3 if not trainable
- run: cd backend && uv run orinth test "$MODEL_ID" --dataset "$DATASET_ID" \
         --fail-under accuracy=0.90 --json > metrics.json
```

## Deferred

- **A standalone `orinth-cli` distribution.** A machine that only talks to a remote Orinth should not
  install TensorFlow to do it. This needs a second Python distribution and a shared schema package,
  and nothing in this design blocks it — the CLI already imports nothing from the backend.
- **Shell completions** for bash/zsh/fish. Cheap with `argcomplete`, but that is a dependency, and the
  command surface should settle first.
- **`--link` ingest**, where a same-machine CLI hands the server a path instead of bytes. Contingent
  on the upload measurement above.
- **Streaming job progress** (SSE or websocket). The UI polls; the CLI polls. Changing that is a
  server change and belongs with whatever else wants it.
- **`orinth recipe`, `orinth chat`, `orinth serve-model`, `orinth arch`.** Phases 11, 15, and 17 have
  API surface a CLI could drive. Left out so this phase's surface stays the one the goal names:
  prep, train, test, infer.
- **A `data_prep_jobs`-backed `orinth dataset prep ls`.** Phase 21 defers that table; without it there
  is no prep job history to list, and the CLI must not invent one.
- **Auth for a non-loopback `--backend`.** Blocked on the platform having auth at all.
