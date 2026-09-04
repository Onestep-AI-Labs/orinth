# Spec: Phase 22 Notebooks

## Status

**Implemented.** The runtime, the proxy, notebook storage, the `orinth` package, the four
templates, and both frontend surfaces are in and exercised against a real `jupyter-server`.

Landed:

- **Runtime** — `services/notebooks/runtime.py` spawns `python -m jupyter_server` on the
  8700–8799 band with `base_url=/api/notebooks/proxy/`, provisions an `orinth` kernelspec into
  `storage/notebooks/.jupyter`, reaps an orphan from a previous process, and stops in the app
  lifespan beside `serving_service.shutdown()`. The presence check is `importlib.util.find_spec`,
  so the API process never imports `jupyter_server` even to ask whether it exists.
- **Proxy** — `services/notebooks/proxy.py`, HTTP and WebSocket, no URL rewriting, token
  injected outbound only.
- **Storage + CRUD** — `services/notebooks/service.py` and `runs.py`; `routers/notebooks.py`
  with the literal paths declared ahead of `/{notebook_id}`.
- **The SDK** — `backend/orinth/`: `datasets` (list/get/readiness/load/records/paths/images/
  register), `models` (list/get/path/artifacts/predictor), `projects`, `settings`, `runs`.
  `PlanSource` gained `"notebook"`.
- **Frontend** — `/notebooks` and `/notebooks/[notebookId]`, `kernel-client.ts` over
  `@jupyterlab/services`, a CodeMirror 6 cell editor themed from platform tokens, the MIME
  dispatch with its allowlist, an ANSI parser, the runtime banner, and a runs rail.

Verified live:

- The runtime starts on port 8700 and offers the `orinth` kernelspec.
- A kernel started **through the proxy** reaches `import orinth` and sees all 16 datasets in the
  workspace.
- The round trip works: a 60-row polars frame registered as `text_classification` came back
  `Ready to train` with a 42/12/6 split and labels derived from the label column, and
  `load()` read it back with the documented columns.
- `load()` refuses a split over the cap (10,170 rows against a 100 cap) naming `limit=` and
  `records()`; `limit=` then works.
- Runs write `run.json` plus an append-only `metrics.jsonl`, artifacts land under `artifacts/`,
  and a non-numeric metric is refused pointing at `log_text`.
- **No token or auth header appears in any proxied response.**

Three defects were found by running it, none visible from reading the code:

- `orinth.datasets` shadows the builtin `list`, so `_ref` calling `list(...)` internally raised
  `TypeError` on the first successful register. The module now uses `builtins.list` explicitly.
- The plan `register()` builds names its mapping decision `field_mapping`, but `DecisionField`
  calls it `mapping` — validation failed at `/prep/apply`, at the far end of an upload.
- Unrelated but real: `format_io._read_annotation_json` called `.get` on its payload *before*
  the `isinstance(payload, list)` fallback could pick it, so a bare-list annotation sidecar
  raised `AttributeError` instead of being read. The list branch was unreachable. Fixed with a
  test covering both shapes.

**Follow-up pass.** Markdown cells and the dataset rail landed after the first cut, along with a
defect the first cut introduced: the editor filtered `cell_type === "code"` on load and wrote every
cell back as `code` on save, so opening a template dropped its prose *and the next save erased it
from the file*. That is data loss, not a missing feature — every template is 3–4 markdown cells.
Cells now round-trip their kind, markdown renders by default and edits on double-click, and
`execution_count`/`outputs` are omitted from markdown cells so the file stays valid nbformat in any
other reader. Verified by reading a template through the proxy, projecting it through the editor's
cell model, saving, and reading back: 3 markdown cells in, 3 out, prose intact.

**Editor pass (phase 24 follow-up).** The first cut shipped `python()` for parsing with **no
`HighlightStyle`**, so the parser built a syntax tree nothing consumed and every cell rendered in
one flat ink colour. `syntax.ts` now maps eight token classes onto the `--label-*` ramp — the
sanctioned §2 exception, since colour here encodes what a token *is*. Alongside it: completion from
the kernel's own `complete_request` (verified against a live namespace — after
`df = orinth.datasets.load(…)`, typing `df.sh` offers `shape, shift, show, shrink_to_fit`, which no
static word list could know), Shift-Tab inspect through `inspect_request` with the ANSI stripped
off the docstring, bracket matching and auto-close, 4-space `indentUnit` per PEP 8, and the
standard notebook operations — run all (sequential, stopping at the first error, because a notebook
is a script), clear outputs, move up/down, insert, delete.

**Templates and defect pass (this change).** Six more templates, and four defects that only came
out of driving the real thing in a real browser:

- **`orinth.runs` never worked in a kernel.** `_workspace.notebook_id()` read `ORINTH_NOTEBOOK_ID`
  and *nothing anywhere set it* — one `jupyter-server` serves every notebook, so there is no point
  in the lifecycle where a per-notebook variable could be written. Every `runs.start()` /
  `runs.log()` in a real kernel raised, and the Runs rail could never fill. It now falls back to the
  working directory, which `jupyter-server` sets per session to the notebook's own directory, and
  confirms the identity against `storage/notebooks/<id>/manifest.json` rather than trusting `cwd`.
  Verified in a kernel started through the proxy: cwd is the notebook directory, `notebook_id()`
  resolves, two logged points come back through `GET /api/notebooks/{id}/runs`.
- **A leaked kernel client polled a stopped runtime forever.** `NotebookKernel` constructs its
  managers before `connect()` resolves, and they poll `api/kernels` from that moment. A failed
  connect — or navigating away mid-connect — left the poller running, which against a stopped
  runtime is `GET /api/notebooks/proxy/api/kernels → 503` on an endless backoff from a page nobody
  is on. The connect now lives in an effect that owns and disposes everything it creates, both
  managers are disposed (not just the session), they run `standby: "when-hidden"`, and the
  managers' `connectionFailure` signal re-asks `/runtime` so a runtime that stops under an open page
  turns the banner back to **stopped** instead of retrying in silence. Measured in headless Chrome:
  0 kernel polls and 0 errors in the 12s after navigating away; stopping the runtime under an open
  page now costs 2 requests and settles, with the banner reading `STOPPED · Start runtime`.
- **A websocket to a stopped runtime answered with a malformed 503.** Raising `HTTPException` from a
  websocket route makes Starlette write a denial *and* the exception middleware write another, so
  the response went out with two `Content-Length` and two `Content-Type` headers — which Node's HTTP
  parser rejects outright (`HPE_UNEXPECTED_CONTENT_LENGTH`), so the dev proxy in front of the app
  could not even relay the refusal. It closes before accept now, which is a well-formed handshake
  rejection. `test_a_websocket_to_a_stopped_runtime_closes_instead_of_raising` fails without the fix.
- **The proxy pinned the kernel subprotocol upstream** regardless of what the client negotiated, so
  a client that asked for no subprotocol got JSON framing on one leg and v1 binary framing on the
  other — a silent hang. Both legs now speak whatever the client chose. Verified against both: a
  JSON-text client and a `v1.kernel.websocket.jupyter.org` binary client each execute a cell through
  the proxy.

Alongside them, three things that were working as designed and reading as broken:

- **`/notebooks` was not in the project area**, so the one page whose whole point is *this project's*
  datasets showed the global sidebar with no project switcher. One entry in `isProjectArea`.
- **The dataset rail looked permanently busy.** The catalog costs seconds to build (it walks every
  split of every dataset) and the rail refetched it on the page default, so every visit spent those
  seconds on a spinner — and showed "No datasets here yet" while the first request was still out.
  The rail now passes its own `staleTime` (5 minutes; mutations still invalidate the key), and
  follows the loading contract: skeleton when there is nothing yet, inline spinner only over
  existing rows.
- **The templates are ten, and grouped.** `NotebookTemplate` gained `category`; the service orders
  by `CATEGORY_ORDER` (Start → Data → Vision → Text → Training → Testing → Inference) and the picker
  splits the list at each change. Every new template was executed end to end against the real
  workspace before shipping — the training one trains, the evaluation one scores 0.933 on
  `reference_yolo/valid`, the batch one writes a 45-row CSV artifact.

**The SDK grew three modules, and the picker grew tabs (this change).** The
package covered *data* — read a dataset, read a model, log a run — and stopped
where the platform's actual work starts. A notebook could describe a training
run but not start one.

- **`orinth.train`** — `options()`, `compute()`, `start()`, `get()`, `list()`,
  `cancel()`, `wait_for()`. Starts the platform's own job, so a run begun in a
  cell appears on `/training`, registers a model the catalog knows about, and
  outlives the tab. `wait=False` by default (the opposite of
  `datasets.register()`): training is minutes to hours and blocking a cell on it
  costs the kernel for the duration. A failed run is *returned*, not raised — it
  is a real outcome with metrics attached, and a cell comparing three runs must
  not lose two of them to the one that diverged.
- **`orinth.evaluate`** — `datasets()`, `start()`, `compare()`, `get()`,
  `list()`, `per_item()`, `wait_for()`. `compare()` goes through the API's batch
  route rather than looping `start()`, because that is what groups the jobs under
  one `comparison_id`; a loop produces N unrelated jobs the Testing page cannot
  line up. `per_item()` is the cell that makes a bad aggregate actionable.
- **`orinth.inference`** — `predict()`, `list()`, `get()`. The recorded
  counterpart to `models.predictor()`: slower per call, and the answer gets an
  id, an overlay, and a row on `/inference`. Exactly one of `image=` or `text=`,
  because they are different predictors and guessing is not the API's job.

Types: `TrainingRun`, `Evaluation`, `Prediction`, frozen like the rest, each with
`done` / `__bool__` so `if run:` means *finished and produced something* rather
than *terminal*.

Verified against a live backend, not mocked: `train.start()` queued a real job
that ran three epochs and registered `trained_keras_classification_a5a0bb95`;
`evaluate.start()` scored 0.8 on ten items and `per_item()` returned all ten;
`evaluate.compare()` ranked two models in one comparison; `inference.predict()`
came back with an overlay URL for an image and an `nlp_result` for a string.
Every artifact created that way was deleted afterwards.

**Templates: 10 → 22, three per category minimum.** Ten in one flat list was
already a wall; the tabs below are what made more of them useful rather than
worse. Added: **Tour the orinth SDK** and **Workspace report** (Start), **Merge
two datasets** (Data), **Review detections against truth** and **Find duplicate
images** (Vision), **Inspect an instruction dataset** and **Probe a text model**
(Text), **Train on the platform** (Training), **Evaluate on the platform** and
**Error analysis** (Testing), **Recorded predictions** and **Pick a confidence
threshold** (Inference). **Compare two models** was rewritten — the first cut
built a `rows` list and never called a predictor, so it demonstrated nothing.

`test_the_sdk_modules_are_each_demonstrated_by_a_template` is the rule made
executable: eleven public calls, each of which must appear in some shipped
template. A module no notebook opens with is a module nobody finds.

**The picker is tabbed and paged.** `All` plus one tab per category, each with a
count, six templates to a page; the notebook list pages at nine. Tabs rather than
stacked sections because categories are *alternatives* — you want a training
notebook or a data one — and a tab bar says that while a stack of headings makes
you scan every one to find out. The category list is derived from the order the
backend sent, so a template naming a new category gets a tab with no frontend
change. `Pager` renders nothing at all for a single page: a disabled pager under
six cards is chrome asserting there is more.

Verified in headless Chrome: eight tabs reading `All 22 · Start 3 · Data 3 ·
Vision 3 · Text 3 · Training 3 · Testing 4 · Inference 3`, page `1 / 4` stepping
to `2 / 4` with a different six, the Training tab showing three cards and no
pager, and the notebook list showing `1 / 2` over twelve notebooks.

**Opening, naming, watching, and choosing a machine (this change).** Five things
the notebook surface was missing, and the SDK call that made "your own data"
real.

- **The runtime is asked for, not announced.** Opening a notebook with the
  runtime stopped left a banner and a page of inert cells: every Run button
  disabled, and the reason a status line you had to notice. It is a dialog now,
  shown once per visit — dismissing it leaves the banner for a second try,
  because a modal that returns after you dismissed it is insisting rather than
  asking.
- **A machine dropdown.** `GET /api/notebooks/runtime/targets` flattens phase
  24's two sources — the per-framework device probe and the provider registry —
  into one list: `Automatic`, each probed device, `CPU only`, and the declared
  remote providers **listed and disabled** with their first requirement as the
  reason. `POST /runtime/start` takes the choice and the runtime injects
  `ORINTH_DEVICE` plus `CUDA_VISIBLE_DEVICES` into the server every kernel
  inherits. It is offered only while stopped, and starting on a different device
  while running is refused *by name* — a kernel's environment is fixed at spawn,
  so a live switch would be a control that silently does nothing.
- **Rename, in both places.** The header title is a button that becomes a field;
  the list gains a Rename row. `useRenameNotebookMutation` already existed and
  nothing called it.
- **The running cell is marked and followed.** An accent left edge and a tinted
  gutter (not a fill — the output is what you are trying to read), and
  `scrollIntoView({block: "center"})` as each cell starts, so Run all reads as a
  walk down the notebook. It stops *on* a failure rather than leaving the
  viewport wherever it was, because that is how a traceback goes unread. Dropped
  to `behavior: "auto"` under `prefers-reduced-motion`.
- **`orinth.datasets.upload(path)`** — your own files as a dataset. `register()`
  takes rows in memory; this takes what is on disk, which is how data actually
  arrives. Nothing about the files is declared: the upload is staged with its
  relative paths and handed to **the same prep agent a browser upload goes to**,
  so a dataset made from a cell and one made by dragging the folder in are the
  same dataset. `detect()` is the read half, for looking before applying. Sent
  in batches of 200 files, so a large folder never has to fit in one request.

Four more templates, in a new **Pipelines** category: **Image: folder to
prediction**, **Text: table to prediction**, **LLM: records to a fine-tune**, and
**Upload your own data** (Data). The three pipelines are the whole loop in one
notebook — upload, train, evaluate, predict, log — with every call an `orinth`
call. Each points at a path you supply and falls back to a small borrowed sample
so it runs anywhere. 26 templates now, and the picker's tabs are what made more
of them useful rather than worse.

Verified in a browser and against a live backend: the dialog lists six machines
(three remote, disabled), starting on `cpu` gives `runtime.device == "cpu"`,
rename from the header persists, a running cell is highlighted and cleared, and
Run all walked the page 4,406px following cells 1 → 3 → 5 → 9. The image
pipeline uploaded a folder, trained a real model, scored it at 0.889, and
predicted at 0.927 confidence; the text pipeline did the same from a CSV.

One sharp edge found by running it, now documented where it is read:
**`load(..., limit=N)` is a head, not a sample.** Items come back in directory
order, so `limit=90` on a six-class image set is ninety images of the first
class — which the prep agent then (correctly) refuses to label.

**A notebook that reads like a notebook (this change).** The surface worked and
the chrome did not: the runtime was controlled from the page you are not using
it on, the dialog could be dismissed onto a dead editor, adding a text cell gave
you an empty rendered block, and every cell action was crammed into the gutter
beside the one button anyone presses.

- **The runtime moved into the notebook.** The list page shows nothing about it
  now — you start a runtime because you are about to run something, and that is
  in the editor. `RuntimeMenu` is a trigger carrying the state (`● auto`) over a
  panel with the machine picker and Start / Stop / **Restart**. Rare controls
  fold away; the information they are about stays visible.
- **`POST /api/notebooks/runtime/restart`.** One call rather than stop-then-start
  from the client, because the interesting case is *changing the device*: two
  calls leave a gap another tab can start the old one in.
- **The start dialog cannot be dismissed.** No close, no Escape, no backdrop
  click, and the body does not scroll behind it. Dismissing left the user on a
  notebook where every cell was inert — a page that looks like an editor and is
  not one. The honest choice is not "keep the dialog or not" but *start it, or
  leave*, so it offers exactly those: **Start runtime** and **Back to notebooks**.
- **A Colab-shaped toolbar.** One row of named verbs — `+ Code`, `+ Text` |
  Run all, Interrupt | Restart kernel, Clear outputs — with everything that
  resets state on the far side of a divider from everything that does not. Text
  buttons, not icons: a toolbar you have to guess at is not simpler, it is
  quieter.
- **Cell chrome, rearranged.** The gutter holds the run button and `[3]` and
  nothing else; move / edit / delete became a hover toolbar at the cell's
  top-right; and inserting a cell happens *in the gap between two cells*, which
  is where you are looking when you want one. All of it is in the DOM at rest
  and revealed on hover or focus, so keyboard users reach it by tab and a
  notebook at rest is cells and nothing else.
- **Text cells get a formatting toolbar and a live preview** — heading, bold,
  italic, code, link, image, quote, lists, rule, LaTeX, table, and Close —
  because markdown is a language people half-know, and someone who cannot
  remember whether a link is `[]()` or `()[]` should not have to leave to find
  out. Actions act on the *selection*: wrapping wraps it, prefixes toggle across
  every line it touches. A **new** text cell opens in the editor while one
  **loaded from a file** renders, which is the distinction that matters: you
  pressed Text in order to write some.
- **Templates open in a modal.** The list page is now `New notebook` and
  `Open a template · 26`; the tabs, the pager and the name field moved inside it.
  Twenty-six cards permanently occupying the page above your own notebooks had
  the ratio backwards.

**And the dataset rail's constant refreshing had a cause, on disk.** The catalog
query polls every four seconds while any dataset reads `detecting` / `planning` /
`applying` — and a prep state is a string on a manifest with no liveness behind
it, so a backend killed mid-run leaves one `planning` **forever**. This
workspace had one from a crash weeks earlier, which meant every page mounting
the rail re-read the whole catalog (a walk of every split of every dataset)
every four seconds, indefinitely. Two fixes, both needed:

- `DatasetPrepService.reconcile_stale_preps()` at startup, beside
  `reconcile_stale_jobs` and `reconcile_stale_recipes` and for the same reason:
  nothing this process did not start is running.
- The rail passes `poll: false`. It lists ids to copy; a prep run's progress
  belongs to the Datasets page.

Verified in a browser: the list page carries no runtime UI and two buttons; the
template modal shows nine tabs and pages 1/5; the start dialog survives Escape
with the body locked and no close button; the runtime menu opens on six
machines with Restart and Stop; `+ Text` opens an editor with twelve toolbar
buttons and a preview, and Bold over a selection produces `**heading text**`
that renders as `<strong>` on Close.

**Four follow-ups from using it.**

- **The start dialog never closed.** It opened from an effect and closed from a
  handler — and once the dialog lost its dismiss (there is nowhere to dismiss it
  *to*), nothing set that flag back, so it sat over a perfectly running kernel.
  Visibility is now *derived*: the dialog is exactly "the runtime is not up".
  Starting one closes it because the condition stops being true.
- **Modals could not scroll.** `.modal-panel` is `overflow: hidden` so its
  corners and header stay put; without `overflow-y: auto` on `.modal-body` a
  long dialog was simply clipped and its bottom unreachable.
- **Modal sizing is fixed, not content-sized.** Switching a template tab from
  six cards to three resized the dialog and moved its own buttons under the
  cursor. The panel now has a height; the body scrolls inside it. The per-dialog
  modifiers are written `.modal-panel.nb-template-modal` — as single classes
  they tied with the base rule and lost on source order, which is how the wide
  template dialog rendered at 560px.
- **The machine list was empty for seven seconds.** `GET /runtime/targets`
  spawns the device probe, measured at **7.1s cold** and 1.7ms warm, because it
  imports torch and TensorFlow in a subprocess — the right place for it, and the
  reason the dropdown had nothing in it on a cold backend. The query now carries
  `Automatic` as `placeholderData`: not a placeholder for a real device but *the
  default*, and the one choice correct on every machine, so the picker works
  from the first paint with a caption saying the rest is still being found.

**Deferred, and not attempted:** `ipywidgets` and interactive output (out of scope by design);
`text/html` output rendering — it falls through to `text/plain` with a note, because sanitizing
arbitrary kernel HTML needs DOMPurify and a policy, and a half-sanitized
`dangerouslySetInnerHTML` is a script-injection hole in the studio; and headless execution.
Everything above is measured; nothing in this section is projected.

## Goal

Give the user a real Jupyter notebook inside Orinth, on the same machine and in the same Python
environment as the platform, with the platform's data one import away.

Today the escape hatch out of Orinth is a manual export: to look at a dataset in pandas, or to try a
model on a valid split before committing to a training run, a user has to find `storage/datasets/`
by hand, work out the on-disk layout, and keep a second environment alive. Whatever they build there
cannot come back — a cleaned dataframe becomes a training set only by re-uploading a file and
letting the prep agent re-derive what the notebook already knew.

After this phase: `/notebooks` holds notebooks scoped to a project, cells run against the backend's
own interpreter, and `import orinth` reaches every dataset, model, and run in the workspace. A
notebook that produces a cleaned dataframe registers it back as a dataset that phase 21's readiness
contract reports as **Ready to train**, without a file leaving the machine.

## Scope

In:

- A managed `jupyter-server` subprocess supervised by the backend, reached only through a FastAPI
  proxy on loopback.
- Notebook lifecycle: create (blank or from a template), list per project, rename, duplicate, delete.
- A notebook editor surface at `/notebooks` and `/notebooks/{id}`: cell list, execution, kernel
  status and control, output rendering, and a dataset/model/run rail.
- The `orinth` Python package, importable in a kernel, covering datasets, models, projects, runs,
  and — through the platform's own job APIs — training, evaluation, and inference.
- Round-trip registration: a dataframe or row list becomes an Orinth dataset through the phase-21
  ingest + apply path.
- Lightweight run logging (`orinth.runs`) with a per-notebook metrics view.
- Twenty-two starter templates, tabbed by category and paged, that double as the SDK's
  documentation.

Out:

- **Multi-user anything.** No per-user kernels, no quotas, no isolation between notebooks beyond the
  fact that each kernel is its own process. See Security.
- **Sandboxing the kernel.** Deliberate, argued in Security.
- **ipywidgets and interactive output.** The widget manager is a Lumino runtime, not a MIME type;
  admitting it means admitting JupyterLab's rendering stack. Deferred.
- **Headless / scheduled notebook execution** (`nbclient`, "run this notebook nightly"). That is a
  job-runner feature and needs the job table and history this phase does not add.
- **Terminals.** `jupyter-server`'s `terminado` handler is not exposed by the proxy.
- **Editing the repo-root `notebooks/` directory.** See "Where notebooks live".
- **Changes to any training, evaluation, or export runner**, and to the prep agent. The round trip
  posts to endpoints phase 21 already ships.
- **A DB table.** No Alembic migration in this phase; argued below.
- Real-time collaboration (`jupyter-collaboration` / Yjs). Two tabs on one notebook is an edge case,
  not a feature.

## Decisions

### Runtime: `jupyter-server` as a managed subprocess, headless, proxied

**Picked: (a), narrowed.** The backend supervises one `jupyter-server` process bound to
`127.0.0.1` on an allocated port; every browser request reaches it through
`/api/notebooks/proxy/…`. **`jupyterlab` is not installed and its UI is never served** — only
`jupyter-server`'s HTTP + WebSocket API (`/api/contents`, `/api/sessions`, `/api/kernels`,
`/api/kernelspecs`) is used.

Why this wins on the repo's actual constraints:

- **Heavy ML imports stay out of the FastAPI process.** `docs/ai/rules.md` forbids importing
  TensorFlow, Ultralytics, OpenCV, or PyTorch at module import. A kernel is a separate process
  spawned by a separate process, so a cell that does `import torch` costs the API nothing — the same
  posture the training runners and the llama.cpp server already have. The backend never imports
  `jupyter_server` either; it spawns `python -m jupyter_server` and probes for the package with
  `importlib.util.find_spec`, which does not execute it.
- **It works offline in the desktop build.** `jupyter-server` is a local wheel and the client is
  bundled by Next. Nothing fetches anything at runtime.
- **The subprocess supervision pattern already exists here.** `app/services/serving.py` allocates a
  loopback port, spawns, probes readiness, monitors, culls on idle, records crashes, reaps orphans by
  PID, and shuts down on lifespan exit. The notebook runtime is that service with a different binary,
  and it gets to *delete* the idle-cull loop because `jupyter-server` has one
  (`MappingKernelManager.cull_idle_timeout`) and llama.cpp does not.

Why **(b) `jupyter_client` + a raw ZMQ kernel** loses: it is the kernel half only. The half we would
still have to build is a ZMQ↔WebSocket bridge inside FastAPI that multiplexes the shell, iopub,
stdin, and control channels, HMAC-signs every message with the session key, correlates
`execute_reply` to `execute_request`, buffers iopub for a client that reconnects mid-cell, and
implements interrupt and restart semantics. That is the Jupyter messaging protocol, and what we
would produce is a worse `jupyter-server` living in the process that serves the UI. It also buys
nothing the pick does not already have: the kernel is a subprocess either way. The one genuine
advantage — no second HTTP server — is not worth owning a protocol.

Why **(c) Pyodide** loses, decisively: the feature is reading the workspace off local disk. Pyodide
runs in the browser sandbox with no access to `storage/`, so every dataset would have to be streamed
over HTTP into WASM memory — the exact cost the "read direct, write through the API" rule below
exists to avoid. There are no Pyodide wheels for `torch` or `ultralytics`, so
`orinth.models.predictor()` could not exist, and `tensorflow` is out of reach. It would also make
the notebook the one place in Orinth that cannot run the platform's own code, which inverts the
point of the feature.

### Where notebooks live

`storage/notebooks/` — ignored, per `AGENTS.md`'s "generated uploads, overlays, DB files, logs, and
training artifacts under ignored `storage/`".

```
storage/notebooks/
  .jupyter/kernels/orinth/kernel.json    # the workspace kernelspec (see Edge Cases)
  <notebook_id>/
    notebook.ipynb                       # owned by jupyter-server's contents manager
    manifest.json                        # Orinth's record: id, project_id, name, tags, timestamps
    .ipynb_checkpoints/                  # jupyter-server's own checkpoints
    runs/<run_id>/{run.json, metrics.jsonl, artifacts/}
    outputs/                             # the kernel's cwd; whatever a cell writes
```

`ServerApp.root_dir` is `storage/notebooks/`, so the contents API and the notebook UI address a
notebook as `<notebook_id>/notebook.ipynb`. The kernel's cwd is the notebook's own directory, so a
cell that writes `out.csv` lands somewhere scoped and deletable rather than in the repo root.

**The repo-root `notebooks/` directory is never touched.** `AGENTS.md` names it as a read-only
research asset ("Do not move, rewrite, or commit files under `datasets/`, `models/`, or
`notebooks/`") and it currently holds `unet_inception/` and `yolo_11/`. It is not listed, not
opened, and not reachable through the contents API, because a notebook editor autosaves — mounting
it read-only would mean the first keystroke in a reference notebook attempts a write to a path the
repo forbids writing. A user who wants one copies it into the workspace.

**No DB table, no Alembic migration.** This follows phase 11 (recipes are filesystem-only) rather
than phase 21 (prep jobs got `data_prep_jobs`). The distinction phase 21 drew was that a prep run
*spends money and must survive a restart*; a notebook kernel by definition does not survive a
restart, and there is nothing to reconcile on boot — a kernel whose process is gone is simply not
running, which is the correct state with no row to correct. The notebook document is a file, the
runs are append-only files, and the only query is "list by project", which a
`storage/notebooks/*/manifest.json` glob answers exactly as `DatasetService._locations()` already
does for datasets.

Manifests are written with the existing atomic helper (`_write_manifest_json`: sibling temp file,
then rename), not `Path.write_text`. Phase 21 fix 21 is the reason — a reader landing between
truncate and write got a `JSONDecodeError` and the row vanished from the catalog. There must be one
such writer in the codebase, not two.

### The `orinth` package boundary: read direct, write through the API

`orinth` is a **second top-level package** at `backend/orinth/`, added to
`[tool.hatch.build.targets.wheel] packages = ["app", "orinth"]`. The import name in a cell has to be
`orinth`; putting it under `app/` and aliasing through `sys.modules` breaks tab completion,
`inspect.getsource`, and pickling for no benefit. The dependency runs one way — **`orinth` imports
from `app`; `app` must never import `orinth`** — with a test asserting it, because the one tempting
violation (the notebooks service reusing the SDK's dataset readers) inverts the layering and the
service already has `DatasetService`.

The kernel is a sibling process on the same machine with the same environment, so:

- **Reads go straight to disk.** `orinth` calls `app.core.config.get_settings()` and resolves
  `storage_path` itself. The notebook runtime passes `STORAGE_DIR`, `MODELS_DIR`, `DATASETS_DIR`,
  and `DATABASE_URL` into the `jupyter-server` environment — the same four the phase-18 supervisor
  already passes to the backend — so a kernel resolves the identical paths the API does, in the
  packaged app as well as in `make dev`.

  Routing reads through HTTP instead would funnel thousands of file reads per `load()` through the
  same uvicorn worker pool that serves the UI and runs prep jobs. Phase 21 fix 19 is the measured
  version of that mistake: building one page of items cost 15,000 file reads and 4.3 s per request,
  against the pool a prep run was using.

- **Writes go through `POST /api/…` on loopback.** Anything that changes platform state — registering
  a dataset above all — must run the service's validation, the project task gate
  (`_require_project_task`), atomic manifest writes, and readiness recomputation. Two processes
  writing dataset manifests concurrently is precisely the class of bug phase 21 fixes 18 and 21
  closed. The base URL comes from a new `ORINTH_API_BASE` setting (default
  `http://127.0.0.1:8000`), passed into the kernel environment. The desktop supervisor gains one
  more env var alongside the four it already sets, using the loopback port it just chose. A
  self-discovery state file was considered and rejected: the backend does not currently know its own
  bind address, and an env var the supervisor already knows how to set is the smaller change.

`orinth/__init__.py` imports only the standard library, `app.core.config`, and `app.schemas`.
`polars`, `PIL`, and anything heavier are imported inside the functions that need them, so
`import orinth` in a cell is instant and a notebook that only lists models never pays for a
dataframe library.

### The editor: a React notebook surface, not an embedded JupyterLab

The cell editor is third-party. The candidates were an iframed JupyterLab and a React surface built
on Jupyter's own client library. **Picked: the React surface** —
[`@jupyterlab/services`](https://www.npmjs.com/package/@jupyterlab/services) (Apache-2.0, maintained
by the Jupyter team, the exact client JupyterLab itself uses) for sessions, kernels, the WebSocket
channels, and the message protocol; CodeMirror 6 for the cell editor; our own output renderer over a
bounded MIME allowlist.

The iframe loses on three counts, the first of which is fatal:

1. **`next.config.mjs` sets `X-Frame-Options: DENY` and `frame-ancestors 'none'` on `/:path*`.**
   `frame-ancestors 'none'` blocks framing even same-origin, so the iframe would not render at all
   without carving a path-prefix exception into the app's frame headers — weakening a security
   header for the one surface that executes arbitrary code.
2. **Double chrome.** JupyterLab ships a menu bar, a file browser, a launcher, and a status bar
   inside an app that already has a sidebar and a page header. The user would navigate two shells.
3. **`frontend/DESIGN.md` forbids the vendor theme.** Phase 17's precedent is explicit — React Flow
   ships only its structural `base.css` and "every visible surface is restyled from platform
   tokens", so no vendor theme enters the app. An iframe cannot be restyled from outside.

The honest cost of the pick is **output rendering**, which is the work JupyterLab would have given
away. It is bounded by an explicit allowlist (see Interfaces → Frontend), and anything outside it
renders as a labelled "unsupported output" row with a download action rather than silently
disappearing. `ipywidgets` is out of scope for exactly this reason: it is not a MIME type that can
be added to the list.

## Interfaces

### API

New domain router `backend/app/api/routers/notebooks.py`, prefix `/notebooks`, registered in
`backend/app/api/routes.py` alongside the existing routers.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/notebooks?project_id=` | List notebooks, newest-modified first. |
| `POST` | `/api/notebooks` | Create blank, or from `template_id`. |
| `GET` | `/api/notebooks/runtime/targets` | Machines a kernel can run on (probe + providers). |
| `POST` | `/api/notebooks/runtime/restart` | Stop and start, on `device`. The only way to move machines. |
| `GET` | `/api/notebooks/templates` | Starter notebooks (metadata only). |
| `GET` | `/api/notebooks/{id}` | Manifest, cell count, kernel state. |
| `PATCH` | `/api/notebooks/{id}` | Rename, retag. |
| `POST` | `/api/notebooks/{id}/duplicate` | Copy as a new notebook. |
| `DELETE` | `/api/notebooks/{id}` | Stop its kernel, delete the directory. |
| `GET` | `/api/notebooks/{id}/session` | `base_url`, `ws_url`, `kernel_name`, `notebook_path` for the client. |
| `GET` | `/api/notebooks/{id}/runs` | Runs logged from this notebook. |
| `GET` | `/api/notebooks/{id}/runs/{run_id}` | One run plus its metric series. |
| `GET` | `/api/notebooks/{id}/runs/{run_id}/artifacts/{name}` | Artifact download. |
| `GET` | `/api/notebooks/runtime` | Runtime status: installed, state, port, python version, kernel count, error. |
| `POST` | `/api/notebooks/runtime/start` | Start the `jupyter-server` subprocess. |
| `POST` | `/api/notebooks/runtime/stop` | Stop it (kills every kernel). |
| `ANY` | `/api/notebooks/proxy/{path:path}` | HTTP pass-through to `jupyter-server`. |
| `WS` | `/api/notebooks/proxy/{path:path}` | WebSocket pass-through for kernel channels. |

Literal paths (`/templates`, `/runtime/*`, `/proxy/*`) are declared **ahead of** `/{notebook_id}`,
for the reason already documented at the top of `routers/datasets.py`: FastAPI matches in
declaration order and a path parameter declared first swallows the literal.

The proxy does **no URL rewriting**. `jupyter-server` runs with
`ServerApp.base_url = "/api/notebooks/proxy/"`, so every URL it generates for itself is already
correct on the outside, and the proxy forwards method, path, query, body, and the WebSocket frames
verbatim. Its only mutation is injecting `Authorization: token <t>` on the way out, so the
`jupyter-server` token stays server-side and never reaches the browser — the same posture the
OpenRouter key has since phase 11. With the token supplied server-side on every request,
`ServerApp.disable_check_xsrf` is set: the browser holds no credential to protect and the XSRF check
would only reject our own proxied writes.

No kernel restart/interrupt endpoints are added — the client reaches `jupyter-server`'s own
`/api/kernels/{id}/restart` and `/interrupt` through the proxy. Duplicating them would create a
second definition of the same operation.

### Schemas

In `backend/app/schemas.py`:

- `NotebookSummary` — `{id, project_id, name, path, tags, cell_count, created_at, updated_at, kernel: NotebookKernelStatus | None, valid: bool}`
- `NotebookCreate` — `{project_id, name, template_id: str | None}`
- `NotebookUpdate` — `{name: str | None, tags: list[str] | None}`
- `NotebookTemplate` — `{id, name, description, task_types: list[TaskType]}`
- `NotebookKernelStatus` — `{state: Literal["starting","idle","busy","dead","unknown"], kernel_id, connections, last_activity}`
- `NotebookSession` — `{base_url, ws_url, kernel_name, notebook_path}`
- `NotebookRuntimeStatus` — `{available: bool, state: Literal["stopped","starting","running","failed"], port, python_version, kernel_count, error, install_hint}`
- `NotebookRun` — `{id, notebook_id, name, params, status, started_at, finished_at, metric_names, artifacts}`
- `NotebookRunSeries` — `{run: NotebookRun, points: list[dict[str, float]]}`

One existing Literal widens: **`PlanSource` gains `"notebook"`** (currently `"llm" | "heuristic"`).
A plan the user wrote in Python is neither a rule nor a model, and phase 21's transparency contract
is explicit that "heuristic output is never presented as model output" — presenting an
author-supplied plan as `heuristic` would be the same lie in the other direction. It is one Literal
value and one badge branch (`neutral`, alongside `heuristic`).

### Settings

In `backend/app/core/config.py`, following the `serving_*` naming:

| Field | Alias | Default | Why |
| --- | --- | --- | --- |
| `api_base_url` | `ORINTH_API_BASE` | `http://127.0.0.1:8000` | How the kernel reaches the write API. |
| `notebook_port_range` | `NOTEBOOK_PORT_RANGE` | `8700-8799` | A band clear of `serving_port_range` (`8600-8699`) so a llama.cpp server and the kernel gateway never collide. Parsed by the same fallback logic as `serving_ports`. |
| `notebook_ready_timeout_seconds` | `NOTEBOOK_READY_TIMEOUT_SECONDS` | `60` | Startup bound. Short, because `jupyter-server` imports nothing heavy. |
| `notebook_kernel_idle_timeout_seconds` | `NOTEBOOK_KERNEL_IDLE_TIMEOUT_SECONDS` | `3600` | Passed to `MappingKernelManager.cull_idle_timeout`. A kernel holding a 6 GB model open overnight is the case this exists for. |
| `notebook_max_output_chars` | `NOTEBOOK_MAX_OUTPUT_CHARS` | `200000` | Per output stream, saved and displayed. Precedent: `MAX_STREAM_CHARS` in `prep/sandbox.py`. |
| `notebook_load_max_rows` | `NOTEBOOK_LOAD_MAX_ROWS` | `200000` | The cap `orinth.datasets.load()` refuses past. Precedent: `MAX_INGEST_ROWS`. |

No `notebook_executor_workers`. The runtime supervises a subprocess; it does not run jobs on a
thread pool, and kernels are `jupyter-server`'s to manage.

### Backend modules

- `backend/app/services/notebooks/service.py` — notebook CRUD over `storage/notebooks/`, manifest
  read/write, template instantiation. Reads `.ipynb` as **plain JSON** for `cell_count`; it never
  imports `nbformat`, so the API process stays free of the Jupyter stack entirely.
- `backend/app/services/notebooks/runtime.py` — the `jupyter-server` supervisor: presence probe, port
  allocation, kernelspec provisioning, spawn, readiness probe, status, stop, orphan reap, lifespan
  shutdown. Modelled on `app/services/serving.py`.
- `backend/app/services/notebooks/proxy.py` — the HTTP and WebSocket pass-through, including the
  refusal to forward to any host that is not the loopback port this process allocated.
- `backend/app/services/notebooks/runs.py` — reading `runs/*/run.json` and `metrics.jsonl` back for
  the API. The *writing* side lives in `orinth.runs`, in the kernel.
- `backend/app/services/notebooks/templates/*.ipynb` — the starter notebooks, tracked in git. They
  are source, not runtime artifacts, so they belong in `backend/` and not in `storage/`. Each one
  carries its own `metadata.orinth` block (`name`, `description`, `category`, `task_types`); there is
  no second registry to drift from the files, and `category` only decides where a template sits in
  the picker, so adding one never means editing a list.

  | Category | Template | What it does |
  | --- | --- | --- |
  | Start | **Blank** | An empty notebook with `orinth` imported. |
  | Start | **Tour the orinth SDK** | One cell per module, and nothing to clean up afterwards. The map every other template is a route across. |
  | Start | **Workspace report** | Projects, datasets, models and runs as tables — including which models nobody has ever evaluated. |
  | Data | **Dataset EDA** | Load, class balance, length distribution, sample rows. |
  | Data | **Register a cleaned dataset** | Load, clean in polars, `register()` back. |
  | Data | **Merge two datasets** | Reconcile two label sets, concatenate, de-duplicate across the seam, register the union. |
  | Vision | **Image stats and augmentation** | Resolution spread, per-class brightness (a leak check), an augmentation preview grid, and a box/polygon size census. |
  | Vision | **Review detections against truth** | Predicted regions drawn over annotated ones, images ranked by how wrong they are, predicted vs annotated region size. |
  | Vision | **Find duplicate images** | Average-hash over every split: exact collisions, near-duplicates, and the ones that leak across the split boundary. |
  | Text | **Text quality and leakage** | Train/valid overlap, duplicates and inconsistently-labelled repeats, length budget. |
  | Text | **Inspect an instruction dataset** | `llm_finetune` records: prompt/response lengths, chat roles, and the conversations that end on the wrong turn. |
  | Text | **Probe a text model** | Hand-written probes grouped by what they test — negation, length, noise — and where confidence collapses. |
  | Training | **Train an image classifier** | A small Keras CNN over a project dataset, one `run.log()` per epoch from a callback, weights saved as a run artifact. |
  | Training | **Sweep a hyperparameter** | Four learning rates, one run each, ranked — one vectorizer adapted once so the arms stay comparable. |
  | Training | **Train on the platform** | `orinth.train.start()` → `wait_for()` → the registered model, with the compute probe and the option catalog first. |
  | Testing | **Compare two models** | Two predictors over one split, diffed item by item — agreement, and who is right where they disagree. |
  | Testing | **Evaluate a model on a split** | Confusion matrix and per-class precision/recall/F1 computed in the kernel, logged as a run. |
  | Testing | **Evaluate on the platform** | `orinth.evaluate.start()` / `compare()` / `per_item()` — the recorded score, and the rows behind it. |
  | Testing | **Error analysis** | A finished evaluation turned into worst classes, confusion pairs, and the confidently-wrong items. |
  | Inference | **Batch inference to CSV** | A whole split through a kernel-loaded predictor, one row per detection, exported as a run artifact. |
  | Inference | **Recorded predictions** | `orinth.inference.predict()` — an id, an overlay, and a row on `/inference` per answer, for images and for text. |
  | Inference | **Pick a confidence threshold** | Predict once at the floor, threshold many times, and read precision against recall instead of taking 0.65 on faith. |

  They are also the SDK's documentation — a user discovers `orinth` by opening one, which is why the
  API surface must stay small enough to fit in them — which
  `test_the_sdk_modules_are_each_demonstrated_by_a_template` now enforces, by asserting that every
  public call appears in some template. Two calls were added *because* the templates needed them
  and importing `app.schemas` from a user's cell is not documentation:
  `orinth.models.parameters(**overrides)` (the options `/inference` sends, with the platform's
  defaults) and `orinth.models.image_label(detections)` (the largest-mask verdict, through the
  platform's own function rather than restated).
- `backend/orinth/train.py`, `evaluate.py`, `inference.py` — HTTP clients for the platform's job
  APIs (`/api/training/jobs`, `/api/testing/jobs`, `/api/inference`). They start the *same* jobs the
  Training, Testing and Inference pages start, which is the point: a run begun in a cell is on the
  page, in the registry, and alive after the tab closes. Fitting a model inside the kernel instead
  produces an artifact nothing recorded, on a process that dies with the notebook.
- Singletons `notebook_service` and `notebook_runtime` in `backend/app/container.py`;
  `notebook_runtime.shutdown()` added to the lifespan teardown next to `serving_service.shutdown()`.

### Frontend surfaces

- `frontend/app/(platform)/notebooks/page.tsx` and `.../[notebookId]/page.tsx` — thin entrypoints.
- `frontend/features/platform/code/` — the Python editor, shared with the architecture studio:
  `python-editor.tsx` (the CodeMirror 6 setup, theme from tokens, 4-space `indentUnit`, bracket
  matching, a diagnostic margin), `syntax.ts` (the highlight style, moved here from
  `features/notebooks/`), and `python-api.ts` (static Keras/torch completion, used only where there
  is no kernel to ask). See `specs/phase-17-model-architecture-studio.md`.
- `frontend/features/notebooks/` — `notebooks-page.tsx` (list + the template modal),
  `runtime-menu.tsx` (machine + start/stop/restart, in the editor),
  `runtime-dialog.tsx` (the undismissable start prompt),
  `markdown-cell.tsx` (render, plus the formatting editor),
  `notebook-page.tsx` (editor shell), `cell-list.tsx`, `cell-editor.tsx` (kernel completion,
  Shift-Enter, Shift-Tab inspect, over the shared editor),
  `cell-output.tsx` (MIME dispatch), `ansi.ts` (SGR → token spans), `kernel-status.tsx`,
  `dataset-rail.tsx`, `runs-panel.tsx`, `kernel-client.ts` (`@jupyterlab/services` wiring),
  `templates.ts`, `hooks.ts`.
- `frontend/lib/api/notebooks.ts`, re-exported from `frontend/lib/api/index.ts` so
  `import { api } from "@/lib/api"` keeps working.
- `frontend/components/app-shell.tsx` — one nav entry at both rail widths, and `/notebooks` in
  `isProjectArea` so the page carries the project sidebar and switcher like every other
  project-scoped route.
- `frontend/app/styles/platform.css` — selectors under a `.nb-` prefix.
- `frontend/DESIGN.md` — one §6 glyph row and one §8 pattern paragraph.

Output MIME allowlist, in dispatch order: `application/vnd.orinth.run+json` (a run summary the SDK
emits, rendered as a metrics card), `image/png`, `image/jpeg`, `image/svg+xml`, `text/html`
(sanitized with DOMPurify), `text/markdown` (the existing GFM renderer from phase 15),
`application/json` (collapsible tree), `text/plain` (ANSI-parsed). Stream outputs (`stdout`,
`stderr`) and `error` tracebacks render as the dark `--shadow-ink` terminal block phase 15 already
established for `.export-log`. Anything else renders a labelled row naming the MIME type with a
download action.

### Storage

Only `storage/notebooks/`, laid out above. No DB change and no migration — a reader looking for the
Alembic line in Validation will not find one, and this is why.

## The `orinth` package

Signatures are the contract. Types are as written; `Split` is
`Literal["unassigned","train","valid","test"]`.

### Handles

```python
@dataclass(frozen=True)
class DatasetRef:
    id: str; project_id: str; name: str
    task_type: str; format: str; labels: list[str]
    splits: dict[str, int]          # split -> item_count
    path: Path                      # dataset root on disk
    readiness: Readiness

@dataclass(frozen=True)
class Readiness:
    state: str                      # ready | needs_prep | needs_input | blocked
    trainable: bool; summary: str; busy: bool
    checks: list[dict]

@dataclass(frozen=True)
class ModelRef:
    id: str; name: str; family: str; task_type: str
    source: str                     # reference | trained | uploaded | promoted
    path: Path | None; available: bool
```

### Datasets

```python
orinth.datasets.list(*, project_id: str | None = None,
                     task_type: str | None = None) -> list[DatasetRef]
orinth.datasets.get(dataset_id: str) -> DatasetRef
orinth.datasets.readiness(dataset_id: str) -> Readiness

orinth.datasets.load(dataset_id: str,
                     split: Split | Sequence[Split] = "train",
                     *, limit: int | None = None,
                     columns: Sequence[str] | None = None) -> "polars.DataFrame"

orinth.datasets.records(dataset_id: str, split: Split = "train") -> Iterator[dict]
orinth.datasets.images(dataset_id: str, split: Split = "train") -> Iterator[tuple[str, "PIL.Image.Image"]]
orinth.datasets.paths(dataset_id: str, split: Split = "train") -> list[Path]
```

**`load()` always returns a `polars.DataFrame`.** One return type, one mental model — `df.head()`
works on every dataset in the platform. What changes per modality is the *columns*, not the type:

| Task | Columns |
| --- | --- |
| `classification`, `object_detection`, `segmentation` | `item_id`, `split`, `path` (str, absolute), `width`, `height`, `annotations` (list[struct: `label`, `label_id`, `bbox`, `polygon`]) |
| `text_classification` | `item_id`, `split`, `text`, `label`, `label_id` |
| `summarization` | `item_id`, `split`, `text`, `summary` |
| `question_answering` | `item_id`, `split`, `text`, `question`, `answer` |
| `llm_finetune` / `instruction_jsonl` | `item_id`, `split`, `instruction`, `input`, `output` |
| `llm_finetune` / `chat_jsonl` | `item_id`, `split`, `messages` (list[struct: `role`, `content`]) |
| `language_modeling` | `item_id`, `split`, `text` |

polars rather than pandas because **polars is already a base dependency** — phase 21 added it to read
Parquet in the prep agent — so the core return type costs nothing new, and adding pandas would put
two dataframe idioms in one workspace. `df.to_pandas()` works where pandas and pyarrow happen to be
installed (they are not base dependencies); `df.to_dicts()` is the no-dependency escape.

**Images never load pixels.** An image row carries a `path`, so a 40,000-image dataset is a
40,000-row frame of a few MB and pixels are pulled per item through `images()`. This is the answer
to "a dataset that will not fit in memory", together with:

- `load()` raises `DatasetTooLargeError` above `notebook_load_max_rows`, naming `limit=`,
  `records()`, and `images()` in the message. Raising rather than truncating: silently returning
  half a training set is the worse failure, and phase 21's `MAX_INGEST_ROWS` set the precedent of a
  cap that says so.
- `records()` streams `data.jsonl` line by line and never holds the file, which is the path any
  `datasets`/`trl` pipeline wants anyway.

**Reading a dataset the prep agent is rewriting.** `load()` raises `DatasetBusyError` when the
manifest reads `metadata.prep.state == "applying"` — the one state that drops and repopulates split
directories. In every other state it reads, and skips an item that vanished between listing the
directory and opening it, which is exactly the fix phase 21 applied to `list_items_page` (fix 18): a
directory listing is a snapshot, and a bulk read must tolerate that.

### Round trip: registering a dataframe back

```python
orinth.datasets.register(
    data: "polars.DataFrame" | "pandas.DataFrame" | Sequence[Mapping],
    *,
    name: str,
    task_type: str,
    project_id: str | None = None,          # defaults to the notebook's project
    format: str | None = None,              # derived from task_type when omitted
    labels: Sequence[str] | None = None,    # derived from the label column when omitted
    columns: Mapping[str, str] | None = None,   # role -> column, e.g. {"text": "body", "label": "y"}
    split: Mapping[str, float] | None = None,   # default {"train": .7, "valid": .2, "test": .1}
    seed: int = 42,
    stratify: bool = True,
    wait: bool = True,
) -> DatasetRef
```

It writes **nothing itself**. The sequence is:

1. Serialize `data` to newline-delimited JSON in a temp file.
2. `POST /api/datasets/ingest` (multipart, one file, `relative_paths=["data.jsonl"]`) → a draft
   dataset with the rows staged at `_staging/`, exactly as a browser upload would leave them.
3. `POST /api/datasets/{id}/prep/apply` with a fully-specified `DatasetPrepApplyRequest`: a
   `DatasetPrepPlan` carrying `source="notebook"`, the stated `task_type` and `format`, a
   `DatasetFieldMapping` built from `columns`, `labels`, a `DatasetSplitConfig` from `split`/`seed`/
   `stratify`, and one `PrepDecision` per field with `source="user"` and a rationale naming the
   notebook.
4. Poll `GET /api/datasets/{id}/prep/status` to a terminal state when `wait=True`, then return the
   `DatasetRef` with its readiness.

The point of routing through *apply* rather than through a new writer is that apply is the code
phase 21 already validated: `_create_layout` → `_update_manifest` → the per-modality item writers →
`update_dataset` → `process_dataset`. A dataset built this way satisfies the readiness contract
because it is built by the same function readiness was written against. Skipping straight to the
prep agent's *detect + plan* would instead ask it to re-derive what the caller already stated, which
is how a mapping gets guessed wrong.

Consequences that fall out of reusing that path, and are correct:

- A project that has not declared `task_type` gets a 409 from `_require_project_task`, surfaced as
  `ProjectTaskNotAllowed` with the settings hint.
- A classification-family task with fewer than two labels is refused against `MULTICLASS_TASKS`, the
  same constant readiness enforces — phase 21 fix 16 made that one constant for exactly this reason.
- `register()` always creates a **new** dataset. Registering into an existing id is not supported:
  rebuilding a dataset out from under a training job that is reading it is the failure phase 21 fix
  18 lived through, and a notebook is the easiest place to trigger it by accident.

### Models, projects, settings

```python
orinth.models.list(*, task_type: str | None = None, source: str | None = None) -> list[ModelRef]
orinth.models.get(model_id: str) -> ModelRef
orinth.models.path(model_id: str) -> Path
orinth.models.predictor(model_id: str) -> "app.ml.predictors.base.Predictor"

orinth.projects.list() -> list[ProjectRef]
orinth.project() -> ProjectRef            # the notebook's project, from its manifest

orinth.settings.workspace() -> dict       # resolved storage/models/datasets paths, api base
orinth.settings.provider_config() -> dict # {"openrouter_model": ..., "openrouter_key_configured": bool}
```

`models.list()` reads `storage/model_registry.json` plus the reference specs — the same source
`ModelRegistry` reads, through `ModelRegistry` itself. `models.predictor()` returns the platform's
own predictor, so a notebook can run precisely what `/inference` runs and diff it; it imports
TensorFlow or Ultralytics on first call, inside the function, in the kernel process, which is
allowed and is the whole reason the kernel is not the API process.

`provider_config()` never returns the OpenRouter key. That is a rule about the SDK, not a security
boundary — see Security.

### Runs

```python
run = orinth.runs.start(name: str, *, params: Mapping | None = None,
                        dataset_id: str | None = None,
                        tags: Sequence[str] = ()) -> Run     # also a context manager
run.log(step: int | None = None, **metrics: float) -> None
run.log_text(key: str, value: str) -> None
run.log_artifact(source: Path | bytes, *, name: str) -> Path
run.finish(status: str = "completed") -> None

orinth.runs.log(step: int | None = None, **metrics: float) -> None   # the implicit run
orinth.runs.list(*, notebook_id: str | None = None) -> list[RunRef]
```

Writes to `storage/notebooks/<notebook_id>/runs/<run_id>/`: `run.json` for name, params, status, and
timestamps; `metrics.jsonl` appended one `{"step": …, "t": …, **scalars}` per line; artifacts under
`artifacts/`. Append-only and one complete line per write, so a kernel killed mid-experiment leaves
a readable partial series rather than a truncated JSON blob.

Module-level `orinth.runs.log(...)` starts an implicit run on first call and finishes it at kernel
shutdown, so the shortest useful thing a user can type is one line.

**Notebook runs are deliberately not `TrainingJob` rows.** That model carries `model_id`,
`dataset_id`, `hyperparameters`, an artifacts contract, a promote path, and cancel-by-PID — a
notebook experiment has none of them, and `reconcile_stale_jobs` would flip every notebook run to
`failed` on the next boot because there is no process to reconcile. Runs stay a per-notebook
artifact, charted on the notebook page and nowhere else.

## Data Flow

```
Browser (/notebooks/<id>)
   │  @jupyterlab/services over  /api/notebooks/proxy/…   (HTTP + WS, same origin)
   ▼
FastAPI  routers/notebooks.py ── proxy.py ──► 127.0.0.1:<notebook port>
   │        (injects Authorization: token …; never forwards it back)
   │
   ├── service.py   storage/notebooks/<id>/{manifest.json, notebook.ipynb}
   ├── runs.py      storage/notebooks/<id>/runs/*/{run.json, metrics.jsonl}
   └── runtime.py   spawns  <sys.executable> -m jupyter_server
                       root_dir=storage/notebooks
                       base_url=/api/notebooks/proxy/
                       JUPYTER_PATH=storage/notebooks/.jupyter
                       env: STORAGE_DIR, MODELS_DIR, DATASETS_DIR,
                            DATABASE_URL, ORINTH_API_BASE
                              │
                              ▼  spawns one kernel per open notebook
                        ipykernel  (cwd = storage/notebooks/<id>/)
                              │
                    import orinth ─┬─ reads  ──► storage/datasets/**, model_registry.json
                                   │             (direct filesystem, via app.core.config)
                                   └─ writes ──► POST $ORINTH_API_BASE/api/datasets/ingest
                                                 POST …/{id}/prep/apply
                                                       │
                                                       ▼
                                          DatasetSummary.readiness → Ready to train
```

## Design

Per `frontend/DESIGN.md`: no new tokens, no shell change beyond one nav entry, no elevation on a
resting surface.

- **Nav.** One new top-level destination, `/notebooks`, glyph `NotebookPen`, placed **after
  Inference** and before the project-settings entry, at both rail widths. It goes last rather than
  mid-list because the rail's order is the journey — datasets → models → training → testing →
  inference — and a notebook is not a stage in it; it is the surface you leave the rails for.
  `NotebookPen` is distinct from every glyph in the §6 table (a shared glyph is a bug per §6), and
  `Notebook` alone was rejected as reading like documentation, which this app no longer has.
- **List page.** `PageHeader` with the same `NotebookPen` glyph, a "New notebook" `Button` primary,
  a flat template gallery reusing the phase-11 `.recipe-template-card` anatomy (hairline `--surface`
  cards, no gradients), and below it a semantic `<table>` in a `.table-wrap` of name / project /
  kernel state / last modified / actions.
- **Editor page.** Header actions: Run all, Interrupt, Restart, and a kernel `Badge` (`ok` idle,
  `info` starting or busy, `neutral` stopped, `fail` dead). A `.nb-grid` two-column layout — the
  cell column beside a collapsible `.nb-rail` holding **Datasets** (the project's datasets with
  phase-21 readiness badges; clicking one inserts `df = orinth.datasets.load("<id>")`, which is the
  bridge the feature exists for), **Models**, and **Runs** (charted with the phase-14
  `MiniLineChart` over the `--label` ramp through `labelColor`, the sanctioned §2 exception). The
  rail collapses below 1024px, following the phase-13 `.training-config-grid` precedent.
- **Cells.** Hairline `--line` borders, no shadow. The executing cell borders `--accent`; an errored
  cell borders `--danger` with its traceback in the `-tint`/`-strong` pair. The execution count
  renders in `--font-mono` `--ink-subtle` on `--surface`, which §9 permits (the prohibition is
  `--ink-subtle` below 18px on surfaces *darker* than `--surface`).
- **CodeMirror ships structural CSS only**, themed from platform tokens — the phase-17 React Flow
  posture, restated: no vendor theme enters the app.
- **ANSI color** in stream output and tracebacks maps onto existing status tokens — red →
  `--danger-strong`, green → `--success-strong`, yellow → `--warn-strong`, blue → `--info-strong`,
  everything else → `--ink`, with bold and dim as weight and opacity. It is not routed through the
  `--label` ramp: §2 scopes that ramp to data visualization, and a traceback is not a chart.
- **Copy stays operational.** No marketing framing on a `(platform)` route, and the medical-context
  rule carries: template notebooks describe measurement, never diagnosis.

## Security

**Executing arbitrary Python is the feature, and the kernel is not sandboxed.** It runs as the user,
with the user's full permissions, on the user's machine: no import allowlist, no filesystem gate, no
resource limits, no network block. A cell can read any file the user can read, open any socket, and
delete `storage/`. This is stated plainly because the alternative — implying a boundary that does
not exist — is the failure mode that matters.

**Why not the phase-21 sandbox.** `backend/app/services/datasets/prep/sandbox.py` runs a
`transform(rows) -> rows` in a subprocess behind a `sys.meta_path` import allowlist, a `builtins.open`
gate scoped to a work directory, CPU/file/descriptor ceilings, a scrubbed environment, and a wall
clock over a new process session. It should keep doing that, and a notebook should not get the same
treatment, because the difference is **authorship and consent**, not danger:

- The sandboxed code is written by a model *this codebase prompted*, from data the user just
  uploaded. The user never saw those twenty lines and never asked for them specifically. Its own
  module says the realistic failure is careless — a stray `requests.get`, a write into the dataset.
- A notebook cell is typed by the person sitting at the machine, who could have typed the same thing
  into `python` at a terminal in the next window. Guarding it protects nobody from anybody.
- And the guards would take the feature with them. The allowlist deliberately excludes `os`,
  `pathlib`, and every third-party dataframe library — which is to say `torch`, `tensorflow`,
  `polars`, `orinth` itself, and file writes. A sandboxed notebook is a notebook that cannot do the
  one thing notebooks are for.

**Where a notebook *is* untrusted: one that arrived from elsewhere.** An `.ipynb` downloaded from a
colleague or a repo is someone else's code, and running it is the same act as `python untrusted.py`.
So: Orinth never executes a cell on open, autorun does not exist, and the create-from-file path
shows every cell's source before the first execution — the phase-17 import-dialog precedent, where
importing a graph imports its custom code and the dialog says so.

**Secrets.** The kernel inherits the backend's environment, which carries `OPENROUTER_API_KEY` and
`HF_TOKEN`. They are **not** scrubbed, and that is a decision rather than an oversight: the kernel
has the user's filesystem, so a variable removed from `os.environ` is readable from `backend/.env`
on the next line, and removing it would be a false assurance rather than a control. What *is*
enforced is narrower and true: `orinth` never returns a key from any accessor —
`settings.provider_config()` reports `openrouter_key_configured: bool` and the model name, mirroring
the phase-11 settings masking — so the SDK is not the leak path.

**The `jupyter-server` token is not user authentication.** It is generated per start, injected by the
proxy server-side, and never sent to the browser: it authenticates *the proxy* to `jupyter-server`,
not *the browser* to Orinth. Anything that can reach `/api` can reach the proxy, and therefore the
kernel. `jupyter-server` binds `127.0.0.1` only and the proxy refuses to forward to any host that is
not the loopback port this process allocated, so the exposure is exactly the exposure `/api` already
has — which today, per phase 9, is none, because there is no auth at all.

**Before any networked deployment.** `docs/ai/rules.md` already says "add auth before multi-user or
network-exposed deployment". This phase makes that the single blocking requirement rather than a
precaution, because the endpoint being exposed is remote code execution as the server user. What
must land first, all of it: real authentication on `/api`; per-user kernels with per-user data roots
(one `storage/` shared between users means every notebook reads every user's datasets); and a
container or VM boundary per kernel — which is the same conclusion `prep/sandbox.py`'s own threat
model reaches for untrusted third-party code. Loopback binding stays regardless; it is a floor, not
a substitute.

## Edge Cases

- **`jupyter-server` not installed** (a developer whose `.venv` predates this phase):
  `GET /runtime` returns `available: false` with `install_hint: "cd backend && uv sync"`, and
  `/notebooks` renders an `EmptyState` naming that command rather than a broken editor. Same posture
  as phase 14's "install the `llm` extras" gate.
- **Kernel crash** (an OOM kill on a large `torch` load, a segfault in a native library):
  `jupyter-server` reports the kernel dead over the WebSocket; the page shows a `danger`-toned strip
  naming the last executed cell and a Restart action. Outputs already streamed are in the document,
  so a crash never loses the log that explains it.
- **Restart.** Clears the namespace, keeps outputs on disk, resets execution counts on the next run.
  An implicit `orinth.runs` run is finished with `status="interrupted"` so its metric series has an
  end rather than trailing off.
- **A runaway cell.** No CPU or wall limit is imposed — see Security. Interrupt sends `SIGINT`
  through `jupyter-server`'s `/api/kernels/{id}/interrupt`; a kernel that ignores it is killed by
  Restart. The real hazard is unbounded stdout growing the document without limit, so each output
  stream is capped at `notebook_max_output_chars` both in the view and on save, with an explicit
  "output truncated" marker. Silent truncation is not acceptable; a marker that says what happened
  is.
- **A dataset that will not fit in memory.** `load()` raises `DatasetTooLargeError` above
  `notebook_load_max_rows` naming `limit=`, `records()`, and `images()`; image frames carry paths and
  never pixels; `records()` streams. Covered in full under the SDK.
- **Reading a dataset mid-prep.** `DatasetBusyError` while `metadata.prep.state == "applying"`;
  vanished items are skipped in every other state.
- **Concurrent notebooks.** One `jupyter-server`, N kernels, one process each — so two notebooks
  never share a namespace. They do share the machine, and two kernels each loading a 6 GB model will
  OOM. There is no quota: the runtime panel reports `kernel_count` and idle kernels are culled after
  `notebook_kernel_idle_timeout_seconds` using `jupyter-server`'s own
  `MappingKernelManager.cull_idle_timeout` rather than a culler of ours — the opposite of
  `ServingService`, which had to write one because llama.cpp has none.
- **Two tabs on one notebook.** `jupyter-server` keys sessions by path, so both attach to the same
  kernel and see the same outputs, which is correct. Saves are last-write-wins; a phase-17-style
  version check is not available because `jupyter-server` owns the save. The second tab shows an
  `info` strip driven by the session's connection count so the collision is visible rather than
  silent.
- **The desktop build, where no system Python exists.** The app's interpreter *is*
  `runtime/backend/.venv/bin/python`, provisioned by `uv python install 3.11` (phase 18). The runtime
  spawns `sys.executable -m jupyter_server` — never `jupyter` off `PATH`, never a system
  interpreter — which is the same resolution phase 18's supervisor already uses for uvicorn.
  Kernelspec discovery is the trap: a user's pre-existing `python3` kernelspec in
  `~/Library/Jupyter` would shadow ours, and the notebook would run in the wrong interpreter with no
  `orinth` and no torch. So the runtime **writes its own kernelspec** to
  `storage/notebooks/.jupyter/kernels/orinth/kernel.json` pointing at `sys.executable`, sets
  `JUPYTER_PATH` to that directory, and pins `MappingKernelManager.default_kernel_name = "orinth"`.
- **Offline.** Nothing fetches at runtime: `jupyter-server` is a local wheel, the client and
  CodeMirror are bundled by Next, and the templates ship in the repo.
- **CSP in the packaged app.** Production CSP is `connect-src 'self'`. CSP3 treats a same-origin
  WebSocket as matching `'self'` and current Chrome and Firefox do, but the desktop build is
  WKWebView and this has not been verified there. The desktop build therefore adds
  `ws://127.0.0.1:* ws://localhost:*` to `connect-src`, gated on `DESKTOP_BUILD=1` exactly as
  `output: "standalone"` already is, and the browser build is left unchanged. This is stated as a
  risk with a manual check in Validation, not as a verified fact.
- **Notebook file corruption.** The `.ipynb` is JSON. If it fails to parse, the list still renders
  the notebook from its manifest with `valid: false` and a `fail` badge — phase 21 fix 21's lesson is
  that an unreadable file must not remove the row — and opening it offers "Open as raw JSON" and
  "Restore checkpoint" from `.ipynb_checkpoints/`. It is never auto-repaired; silently rewriting a
  user's notebook is worse than showing it broken.
- **Notebook deleted on disk while open.** The save returns 404 from the contents API, surfaced in
  the panel and the toast layer per `docs/ai/rules.md`.
- **Backend restart with kernels running.** The lifespan teardown stops the runtime, which
  `SIGTERM`s the process group and `SIGKILL`s after a grace period. On the next boot, orphans are
  reaped by recorded PID *and process start time*, the phase-18 approach that detects PID reuse
  without needing to identify the process any other way; anything unverifiable is left alone.
- **A project with no datasets.** The rail's `EmptyState` links to `/datasets`, per the §8 journey
  rule.

## Dependencies

Backend, added to **`[project.dependencies]` — the base group, not an optional extra**:

- `jupyter-server>=2.14` — the HTTP/WS API. Pulls `jupyter-client`, `jupyter-core`, `nbformat`,
  `traitlets`, `tornado`, `pyzmq`, `jupyter-events`, `argon2-cffi`, `Send2Trash`.
- `ipykernel>=6.29` — the kernel. Pulls `ipython`, `debugpy`, `comm`, `matplotlib-inline`,
  `nest-asyncio`, `psutil`.

`jupyterlab` is **not** added. The pick serves no Jupyter UI, and leaving it out saves roughly 80 MB
of static assets that would ship in the `.dmg` for nothing.

Why the base group, against the phase-18 precedent that put `llm` and `llm-cuda` out of the desktop
build:

- The desktop bootstrap runs `uv sync --frozen --no-dev` with no extras, so anything in an optional
  group does not exist in the product's only distributable build. Phase 18 accepted that for LLM
  fine-tuning, explicitly calling it "a developer-environment feature". A notebook workspace absent
  from the shipped app is not a feature.
- The weight is roughly 35–45 MB on top of an environment already ~2.7 GB and dominated by
  TensorFlow and torch — under 2%, to be confirmed by measuring the resolved environment before and
  after (see Validation).
- Crucially it adds **no compiler requirement**. `llama-cpp-python` was excluded from the desktop
  build because it builds from source on macOS and needs Xcode Command Line Tools; `pyzmq` and
  `argon2-cffi` ship wheels for macOS arm64 and x86_64 and for Linux, so the bootstrap stays a
  download.
- The lazy-import rule holds: the backend never imports `jupyter_server` or `ipykernel`. It spawns
  the first and probes for it with `importlib.util.find_spec`, which resolves a module without
  executing it.

Frontend (`pnpm add`): `@jupyterlab/services`, `@codemirror/state`, `@codemirror/view`,
`@codemirror/commands`, `@codemirror/language`, `@codemirror/autocomplete`, `@codemirror/lang-python`,
`dompurify`. Roughly 500 KB minified in total, code-split behind the `/notebooks` route so no other
page pays for it — to be confirmed against the route bundle after `pnpm build`. The ANSI parser is
written in-repo (~60 lines) rather than taken as a dependency, because a generic one emits its own
color literals and `frontend/DESIGN.md` requires the token mapping described in Design.

## Acceptance Criteria

Behavior:

- From `/notebooks`, a user creates a notebook, opens it, runs `print(1 + 1)`, and sees `2` — with no
  terminal, no manual `jupyter` command, and no configuration.
- `import orinth; orinth.datasets.list()` in a cell returns the same datasets `/datasets` shows for
  that project, with matching readiness.
- `orinth.datasets.load(id)` returns a `polars.DataFrame` whose columns match the table in the SDK
  section, for at least one dataset of each of: image, text, `llm_finetune`, `language_modeling`.
- An image dataset's frame carries file paths and no pixel data, and `images()` yields decodable
  images for the same items.
- A notebook loads a text dataset, filters it, and `register()`s the result; the new dataset appears
  in the catalog with `readiness.trainable == true`, its plan reads source `notebook`, and a training
  run started from `/training` against it resolves `prepared_training_root` and starts.
- `orinth.runs.log(step=…, loss=…)` produces a run that charts on the notebook page.
- A cell raising an exception renders the traceback in the `danger` pair; a matplotlib figure renders
  as an image; an unknown MIME type renders a named download row rather than vanishing.
- Interrupting `while True: pass` returns the kernel to idle; Restart clears the namespace.
- Killing the kernel process externally is reported as `dead` with a Restart action, and no output
  already produced is lost.
- Quitting the backend leaves no `jupyter_server` or kernel process behind.
- The repo-root `notebooks/` directory never appears in the workspace and is never written.

Tests:

- `backend/tests/test_notebooks_service.py` — create / list / rename / duplicate / delete; project
  scoping; manifest written atomically; a corrupt `.ipynb` still lists with `valid: false`; the
  root `notebooks/` directory is not reachable.
- `backend/tests/test_notebook_runtime.py` — the workspace kernelspec is written and names
  `sys.executable`; port allocation walks the configured range; a missing `jupyter_server` yields
  the install hint rather than a 500; readiness probe timeout is reported; stop kills the process
  group; orphan reap matches PID *and* start time.
- `backend/tests/test_notebooks_proxy.py` — the token is injected outbound and appears in no response
  body or header reaching the client; a forward to a non-loopback host is refused; `base_url` makes
  path rewriting unnecessary (a proxied path arrives at the upstream unchanged).
- `backend/tests/test_orinth_sdk.py` — `list`/`get`/`readiness` against fixture datasets; the
  `load()` column contract per task family; image frames carry paths and no pixels; `records()`
  streams; the row cap raises and the message names the alternatives; `DatasetBusyError` while
  `applying`; a vanished item is skipped; `models.list()` matches `ModelRegistry`; `runs.log`
  appends and reads back; **`app` never imports `orinth`**; no accessor returns the OpenRouter key.
- `backend/tests/test_orinth_register.py` — round trip to `readiness.trainable` for
  `text_classification` and both `llm_finetune` formats; a project without the task gets the 409; a
  single-label classification set is refused against `MULTICLASS_TASKS`; the resulting plan reads
  source `notebook` with per-field `user` decisions.
- `frontend/features/notebooks/*.test.tsx` — cell reducer (add, delete, move, execution counts),
  output MIME dispatch including the unknown-type fallback, the ANSI → token parser, kernel-state
  badge tones.

Manual checks:

- Light and dark, no hardcoded color, no `box-shadow` on a resting cell or panel.
- Cell focus, run (`Shift+Enter`), and cell navigation work from the keyboard alone.
- Two tabs on one notebook share a kernel and both show the `info` strip.
- **In the packaged `.dmg`, offline**: open a notebook and run a cell. This is the check that
  confirms the desktop CSP amendment and the workspace kernelspec, both of which are argued above
  and neither of which has been verified.
- Measure and record: the resolved environment size before and after the two new dependencies, the
  `/notebooks` route bundle size, and the wall time from `POST /runtime/start` to a kernel accepting
  its first execution.

## Deferred

- `ipywidgets` and interactive output. Needs the Lumino widget manager, which brings the JupyterLab
  rendering stack the editor decision deliberately kept out.
- Headless / scheduled execution (`nbclient`), which needs a job table and history.
- Terminals.
- Real-time collaboration (`jupyter-collaboration`), which would replace last-write-wins.
- A kernel variable inspector in the rail. Useful, but it needs a comm channel and a serialization
  policy of its own.
- Publishing a notebook as a shareable HTML report (`nbconvert` is already a transitive dependency,
  so this is cheap later).
- Per-kernel resource limits and container isolation, which arrive with the auth work Security names
  as the precondition for any networked deployment.
- Exposing the repo-root `notebooks/` references read-only. Blocked on a contents manager that
  genuinely cannot write, which `jupyter-server` does not offer at a granularity worth trusting.

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd frontend && pnpm generate:api
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm test
cd frontend && pnpm build
```

No Alembic step: this phase adds no DB model and therefore no migration. `pnpm generate:api`
re-exports the OpenAPI document and regenerates `frontend/types/generated/api.ts`, so the new
schemas reach the frontend types without hand-editing.

Plus one end-to-end run, offline: create a notebook from the **Register cleaned dataset** template,
run every cell, and confirm the dataset it produces appears in the catalog as **Ready to train** and
trains for one epoch from `/training`.
