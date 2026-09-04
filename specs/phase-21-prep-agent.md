# Spec: Phase 21 Prep Agent

## Status

In progress.

Implemented:

- **Readiness contract** — `prep/readiness.py`, `DatasetReadiness` on every `DatasetSummary`,
  `GET /datasets/{id}/readiness`, surfaced on the catalog card, the studio header, and the training
  page (which now explains a blocked Start button instead of silently disabling it).
- **Detection** — `prep/detect.py`, with the four `format_io` sniffers and `dataset_hub._detect_format`
  lifted to module level so there is one implementation of each. Verified against all 14 real
  datasets in the repo across every modality; all 14 resolve to the correct task type.

- **Planning** — `prep/plan.py`. The heuristic path is the base case and always runs; the LLM
  refines it when a key is configured, and every failure degrades with a stated reason.
  `services/providers/` ported for the client.
- **Ingest and apply** — `prep/staging.py`, `prep/apply.py`, `prep/service.py`, and eight endpoints
  on the datasets router. Raw files are staged with their relative paths intact, then moved into
  the layout the training runners already read.

- **Dataset Studio restructure** — `Overview | Data | Prepare` replacing the four peer tabs, an
  ingest drop card on the catalog that runs the agent on upload, and readiness on the training
  page. Guided tours and `frontend/DESIGN.md` updated to match.

- **One way in** — `new-dataset-panel.tsx` collapses the four competing create affordances into one
  panel with a source switcher (Files / HuggingFace / Documents / Empty). The catalog toolbar keeps
  only Refresh.

- **Named progress** — `DatasetPrepStatus` gains `step` / `detail` / `progress`, written at every
  stage transition; `prep-progress.tsx` renders the stage ladder with the server's own sentence
  under it, in place of the word "working".

- **Readiness no longer downgrades on a re-run** — analysis is reported through
  `DatasetReadiness.busy` instead of as a state, and `prep_applied` is advisory once the data
  itself is trainable. Asking Orinth to re-read a ready dataset used to make it report "Preparing…"
  and then "Needs prep".

- **One grid for every modality** — `GET /datasets/{id}/table` plus `data-grid.tsx`: a sortable-page,
  editable table (label and split in place) that replaces having to know whether a dataset is images,
  text, or records before you can look at it.

- **Sandboxed transform** — `prep/sandbox.py` and `prep/transform.py`. When planning cannot produce
  a plan that would yield training examples, a Python `transform(rows)` is generated — by an
  OpenRouter model where a key is configured, from column statistics otherwise — and run in a
  subprocess with an import allowlist, no network, no filesystem, and CPU/memory/wall ceilings. Its
  output is validated against the task it claimed and written to `_derived/`, which apply then
  ingests in place of the raw rows. This is what closed the feature-table gap: `uciml/iris` and both
  Kaggle datasets in the table below used to stop at "pick a task manually".

  The gate is *applicability*, not "no task was named" (fix 14 below). A plan naming a task it has
  no columns to feed is the worse case of the two, because it applies: it writes items with no
  annotations and reports success. Both reach the sandbox, so what the transform may produce is not
  restricted to what detection thought the data was.

Pending: the `data_prep_jobs` table and background execution (the flow is synchronous today, which
is fast enough at current dataset sizes — the largest real upload tested took 26 seconds), and the
phase-20 tabular studio, which remains the right home for genuine tabular modelling.

### Validated against external data

The agent was run over 11 datasets downloaded from Kaggle and the Hugging Face Hub in their native
packaging — Roboflow zips, Kaggle archives, Parquet shards, gzipped JSONL — rather than over
fixtures. The first run passed **0 of 11**; detection and planning took it to **10 of 11**, and the
transform stage closed the last one. All of it runs unattended with no OpenRouter key. Everything
in the lists below was found that way.

| Source | Raw shape | Detected | Result |
|---|---|---|---|
| kaggle `uciml/iris` | CSV, numeric columns | *(none)* → transform | 105/30/15, target `Species`, `Id` dropped |
| kaggle `uciml/sms-spam-collection` | CSV, `v1`/`v2` | text classification | 3901/1114/557 |
| kaggle `crowdflower/twitter-airline-sentiment` | CSV + sqlite | text classification | 10247/2929/1464 |
| kaggle `datatattle/covid-19-nlp` | 2 CSVs | text classification | 16620/4750/2374 |
| kaggle `shivamb/netflix-shows` | CSV | text classification | 6165/1761/881 |
| hf `keremberke/chest-xray-classification` | Roboflow zip, class folders | classification | 2/2/0 |
| hf `keremberke/blood-cell-object-detection` | Roboflow zip + COCO json | object detection | 2/1/0, 27 annotations |
| hf `keremberke/pokemon-classification` | Roboflow zip, 35 classes | classification | 35/35/0 |
| hf `tatsu-lab/alpaca` | Parquet, 52k rows | llm_finetune (instruction) | 13992/3998/1999 |
| hf `rajpurkar/squad` | Parquet, nested answers | question answering | 7399/2114/1057 |
| hf `Anthropic/hh-rlhf` | gzipped JSONL, chosen/rejected | llm_finetune (chat) | 1648/471/235 |

The transform stage was built against three **feature tables**, whose column shapes follow the
published schemas of `uciml/iris`, `mansiaggarwal88/diabetes-risk-prediction`, and
`mobeenfatimah/student-exam-performance-and-success-dataset`. All three are **reconstructions**: the
machine this stage was built on has no Kaggle credentials, so the values are generated to match the
documented columns rather than downloaded. The row counts follow from the split ratios and are
right for any file of that size; the accuracies are properties of the generated data and not
measurements of those datasets. Re-run against the real archives before quoting them.

| Table | Shape | Target chosen | Result |
|---|---|---|---|
| diabetes risk | 16 clinical yes/no columns + `class` | `class` (named like an outcome) | 365/104/51, 95.2% validation accuracy from `nlp_tfidf_classifier` |
| student performance | numeric + `result` + `exam_score` | `result` (named), `student_id` dropped | 700/200/100, `exam_score` reported as dominant |
| student performance, no `result` | numeric only | `exam_score` (last column, continuous) | 700/201/99 over `low`/`medium`/`high`, 73.1% accuracy |

One of the three has since been run against the real upload rather than a reconstruction. The
student-exam table in `storage/datasets/student-exam-7825fd26` is 44 columns and 100,000 rows; its
first 3,000 rows, with no OpenRouter key so the builtin engine wrote the transform, resolve to
`text_classification` on target `performance_level` over `High`/`Low`/`Medium`, splitting
2100/600/300 and reporting **Ready to train** in 3.3 seconds. No accuracy was measured. The other
two rows above remain reconstructions.

Fixes found by testing against the repo's own datasets:

1. Two detection rules for data Orinth itself wrote (see Behavior → Detect).
2. A wrapper-folder descent, for uploads that nest the dataset one level down.
3. `_locations` treated an explicit `labels: []` as absent and substituted the reference dental
   classes, so an unlabelled draft inherited `granuloma`/`kista` and read as ready to train.
   Absent and empty are now distinguished.
4. Apply wrote an empty annotation sidecar for every image. `format_io._annotations` returns a
   sidecar whenever one *exists*, checking YOLO only when it does not — so the empty file masked
   the `labels/*.txt` behind it and the repo's 1,619-file dental dataset reported 0 of 747
   annotated images. The sidecar is now written only when it has content.

Fixes found by testing against Kaggle and Hugging Face:

5. **Split ratios were being renormalized by floating-point noise.** `0.7 + 0.2 + 0.1` is
   `0.9999999999999999`, so `_normalize_split_config` rescaled ratios that were already correct.
   The perturbation flipped the tie-break in `_targets_for_count`'s largest-remainder pass, and on
   a small class that is the difference between a valid split and an empty one — which makes an
   image dataset untrainable. Ratios summing to 1 within tolerance are now returned untouched.
6. **Column matching was exact-name only.** Kaggle's airline dataset labels its target
   `airline_sentiment`, the COVID one calls its text `OriginalTweet`, and the SMS spam set ships
   `v1`/`v2`. Matching produced a *half* mapping, which is worse than none because apply then
   skipped every row it could not read. Matching is now name-then-shape: word-boundary matching
   first, then inference from column statistics — the free-text column is the longest, the label
   column is the one that repeats.
7. **Parquet was unreadable.** Most Hugging Face datasets ship no other format, so the agent could
   not see the Hub at all. Read via polars, with nested Arrow values flattened (SQuAD stores answers
   as `{"text": [...], "answer_start": [...]}`).
8. **The detection sample cap leaked into apply**, so a 52,000-row Parquet ingested 20 rows.
   Profiling and prompting are now separate limits: 500 rows read for column statistics, 20 put in
   a prompt. This also fixed label inference, which at 20 rows could not distinguish a closed class
   set from free text — every column has at most 20 distinct values in 20 rows.
9. **COCO uploads landed where nothing looks for them.** `_image_paths` expects COCO images at
   `<split>/` and everything else at `<split>/images/`, so a COCO dataset stored as COCO was
   invisible to the splitter and read as empty. COCO is now converted on ingest into the YOLO
   layout the runner actually trains from.
10. **Preference datasets imported as zero rows.** `{"chosen": ..., "rejected": ...}` fails record
    validation one row at a time. The accepted side is a full `Human:`/`Assistant:` transcript, so
    it is parsed into chat turns and the rejected side dropped — the standard way these become SFT
    data.
11. **Large sources had no bound.** The platform stores one file per item plus sidecars, so a
    41,000-row CSV is ~123,000 file creations and an upload that looks hung. Ingest is capped at
    `MAX_INGEST_ROWS` (20,000) per source and says so in the plan warnings.
12. **Splitting was quadratic.** `_move_item` resolves each item through `_item_path`, which
    re-listed and re-sorted the entire split directory on every call — so splitting *n* items cost
    *n* full directory scans. Uploading a few files at a time never exposed it; the agent ingests
    thousands at once and a 5,500-row CSV ran for minutes without finishing. `_image_path`,
    `_text_path`, and `_record_path` now try the direct path first (the item id *is* the filename)
    and keep the scan only as a fallback. The same CSV now completes in under five seconds.
13. **"Run Prep" was shown to datasets that had already run prep** and stopped deliberately, asking
    the user to repeat an action that would stop in the same place. Readiness now surfaces the
    plan's own `needs_input` reason instead.

Fixes found by running the agent on a 44-column, 100,000-row student-exam table
(`storage/datasets/student-exam-7825fd26`):

14. **A model could name a task for a modality that cannot carry it, and nothing checked.** Asked
    about a CSV of student records, `openai/gpt-4o-mini` answered `classification` — the *image*
    task. `_merge` accepted it (it is a real task, and at confidence 0.30 the structural lock does
    not apply), cleared the `needs_input` that would have stopped apply, and the plan went straight
    past the transform stage, whose only gate was `if plan.task_type`. Apply then ran the image
    importer over a directory with no images: zero items, zero labels, three empty splits.

    Three gates now stand between that reply and a dataset. `MODALITY_TASKS` bounds what the model
    may name by what the scan read the files to be — the model still ranks and names, but a table is
    not images however unsure the scan is about which task the rows support. `blocking_reason`
    checks the merged plan as a whole and reverts to the heuristic when it could not produce a
    single example. And the prompt now offers only the tasks that are both enabled for the project
    and possible for the modality, so the reply that has to be thrown away is not solicited.

15. **The mapping was validated per field and never as a whole.** Asked for nine roles, the model
    filled nine — `text: student_id`, `label: pass_status`, `messages: motivation_level` — and every
    one passed, because each names a real column among the 44. Apply reads two of the nine; the
    other seven only appeared on the Overview screen, where they read as columns Orinth had
    understood. Mappings are now pruned to the roles their task reads, re-derived from the scan when
    the model changes the task (a switch to `llm_finetune` needs the `instruction`/`output` columns
    that a text-classification mapping had no place for), and overlaid on the scan's reading rather
    than replacing it — so dropping one invented column no longer discards the roles detection got
    right and leaves the plan with nothing to annotate rows from.

16. **A model could narrow a classifier to one class.** `plan.labels` assigns class ids, and
    `_row_annotation` falls back to `class_id=0` for a value not in the list — so dropping a class
    that is present in the data does not drop those rows, it relabels every one of them as the class
    that is left. Refused, against the same two-class minimum `readiness.py` enforces, which is now
    one constant (`MULTICLASS_TASKS`) read by both rather than two that could drift.

17. **A stale `_derived/` could be ingested by a later plan.** `_derived/` *replaces* the raw rows
    in apply, and it was cleared only on the branch that returned a clean plan. Widening the
    transform gate in fix 14 made that reachable: a blocked plan that then produced no transform
    kept the previous run's output and ingested it. It is now cleared before either branch, which
    costs nothing — `write_derived` replaces the directory anyway.

Fixes found by running the agent on a real 15,000-row, 19-column diabetes-risk
table (`~/Downloads/diabetes_risk.csv`) through the running dev stack:

18. **The studio's own polling crashed against the dataset the agent was rebuilding.** `/items` and
    `/eda` enumerate a split directory and then open each file in it, while `apply` is deleting that
    directory and repopulating it from staging (`_reset_items`). Between listing a path and reading
    it, the file is gone — and an uncaught `FileNotFoundError` turned into a 500 for the whole
    request, so the console filled with tracebacks for the duration of every prep run. A directory
    listing is a snapshot; a bulk read now skips an item that vanished under it. `item_detail`
    still raises, because a request naming one item wants "gone" reported rather than swallowed.

19. **Paging read the entire dataset to return one page.** `list_items_page` built a full
    `DatasetItemSummary` — opening the item *and* its annotation sidecar — for every item in every
    split, then sliced fifty rows out of the result. On this upload that is 15,000 file reads per
    request, 4.3 seconds, repeated on every poll, against the same thread pool the prep run is
    using. Without a filter the page is decided by the paths alone, so only that window is read now:
    **4.30s → 0.10s**. A filtered page still scans, because which items match is decided by their
    annotations and only reading supplies those.

20. **Prep was a 37-second HTTP request.** Measured end to end
    against a real uvicorn: the backend answers `200` and the dataset is correct
    (`text_classification` on `diabetes_risk`, three classes, 10170/2905/1453) — it is the Next.js
    dev proxy in front of it that gives up and resets the socket, which is the `ECONNRESET` the user
    sees. Nothing is lost; the work completes server-side and a reload shows a prepared dataset.
    Profiling puts 20.2s of 21.7s in `apply`, and it is not one hot spot: ~45,000 file creates
    (`_write_text_item` per row, plus its annotation sidecar) and ~45,000 renames
    (`process_dataset` moving every item out of `unassigned/` into its split). That is inherent to
    one-file-per-item storage, so the fix was not a micro-optimization: **the run moved off the
    request path**, following what recipes, training, evaluation and exports already do here — one
    small executor per job kind (`prep_executor`), `submit_job` for anything that escapes, and the
    frontend polling instead of holding a socket. `POST /prep` now answers **202 in 7ms** with the
    dataset already reading `planning`, and the readiness contract that anticipated this
    (`_PREP_IN_FLIGHT`, "a run in flight beats every other verdict") finally has a run to describe.

    Polling needed a cheap endpoint, not the obvious one: `GET /datasets` builds every dataset's
    split counts and costs 2-3s on this upload, so polling *that* would have recreated the load it
    replaced. `GET /{id}/prep/status` is one manifest read — 4ms — and is what the run is watched
    through; the catalog is refetched only while a run is in flight, and invalidated once at the end.

    A background run has no response to raise into, so `DatasetPrepStatus.error` carries why one
    stopped, and a second run on a dataset already being prepared is refused with 409 rather than
    queued — two runs would interleave one's `_reset_items` with the other's writes.

    Still deferred: the `data_prep_jobs` *row*. A restart forgets a run in flight and leaves its
    dataset reading `applying`. The manifest is the right home for what a dataset *is*; a table is
    needed for what a run *was*, and that belongs with job history.

21. **Manifest writes were not atomic, so a dataset could vanish mid-run.** `_update_manifest` used
    `write_text`, which truncates and *then* writes, while `_locations` re-reads every manifest on
    almost every request. A prep run rewrites its manifest at each state transition, and a reader
    landing in that window got a `JSONDecodeError` — which `_locations` handles by skipping the
    dataset, so it disappeared from the catalog and the request 404'd. This is what surfaced when
    the run moved to a thread and the tests started polling it. Manifests are now written to a
    sibling temp file and renamed, so a reader sees the old manifest or the new one and never
    neither. The regression test fails against the previous implementation.

    With 18, 19 and this in place, a real run was measured with `/items` and `/eda` polled
    continuously throughout: **290 polls across a 40-second run, zero errors**, against the same
    dataset the agent was rebuilding.

## Goal

Let a user drop raw data and get a dataset that is demonstrably ready to train, without declaring
what the data is first.

Today the platform asks for domain, task type, record schema, and class labels *before* it will
accept a file, then hides the action that actually makes a dataset trainable ("Proceed") inside a
collapsed accordion on the fourth of four peer tabs. Nothing anywhere reports whether a dataset can
be trained on; the user finds out when a run fails, or — worse — when it silently succeeds on zero
items.

After this phase: drop files or a folder, the Orinth agent detects the modality, task, format and
labels, profiles the data, plans preprocessing and splits, applies the plan, and the dataset shows
a **Ready to train** state. Every decision the agent made is labelled with its source, rationale,
and the evidence behind it, and the whole thing is reversible.

## Scope

In:

- **`DatasetReadiness`** — a structural, per-task contract computed on every `DatasetSummary`,
  surfaced on the dataset catalog card, the studio header, and the training page's dataset select.
- **No-declaration ingest** — `POST /datasets/ingest` accepts files with no task type, format, or
  labels, preserving each file's relative path in a staging area.
- **The prep agent** — a four-stage job (detect → profile → plan → apply) that turns a staged
  upload into a split, trainable dataset.
- **Deterministic detection** covering YOLO, COCO, image-folder, text-folder, instruction/chat
  JSONL, and delimited-table inputs.
- **LLM-assisted planning** through OpenRouter when a key is configured, with a deterministic
  heuristic fallback that always runs first and always produces a usable plan.
- **A sandboxed Python transform** for data no rule maps — feature tables above all. The script is
  written by a model or generated from column statistics, runs in `prep/sandbox.py`, and is
  validated against the task it claims before its output is used.
- **Decision transparency** — `PrepDecision` records source/confidence/rationale/evidence per plan
  field; `PrepEngine` records which engine ran, why it degraded, and what it cost.
- **Reversibility** — raw files retained in staging, a pre-apply snapshot, and an Undo action.
- **Dataset Studio restructure** — `Overview | Data | Prepare`, one primary action, raw ML knobs
  behind an Advanced disclosure.
- **Training page integration** — readiness-aware dataset selection and one explanatory sentence in
  place of four silent disabled conditions.

Out:

- **Changes to any training, evaluation, or export runner.** The agent produces the on-disk layout
  the runners already read; `prepared_training_root` is unchanged.
- **Changes to the splitter.** `process_dataset` is called as-is.
- **Tabular *modelling*.** The transform stage makes a feature table trainable by serializing it
  into text-classification examples, which is the difference between a spreadsheet the platform
  refuses and one it can train on today. It is not feature typing, imputation, or a gradient-boosted
  trainer — that is `phase-20-tabular-dataset-studio.md`, and this phase must not depend on it
  landing.
- **Auth, multi-user, or per-user quotas** on prep jobs.
- **Providers other than OpenRouter.**
- **OCR and scanned PDFs.** Document ingestion remains `phase-11-data-recipes.md`'s job.
- **`.zip` upload.** Deferred; folder upload covers the case without an extraction guard.
- Readiness-aware selection on the testing and inference pages. Deferred, noted below.

## Interfaces

- API endpoints:
  - `POST /api/datasets/ingest` (multipart) — draft dataset from raw files
  - `POST /api/datasets/{id}/ingest` (multipart) — add raw files to a draft
  - `POST /api/datasets/{id}/prep` — start a prep job
  - `GET /api/datasets/{id}/prep` — latest prep job for a dataset
  - `GET /api/datasets/{id}/prep/status` — run state alone, cheap enough to poll
  - `POST /api/datasets/{id}/prep/apply` — apply a (possibly edited) plan
  - `POST /api/datasets/{id}/prep/undo` — restore the pre-apply state
  - `POST /api/datasets/{id}/prep/cancel`
  - `GET /api/datasets/prep/jobs?project_id=&limit=&offset=`
  - `GET /api/datasets/{id}/readiness`
  - `GET /api/datasets/{id}/table?split=&class_name=&unlabeled=&limit=&offset=` — the grid view
- Schemas (`backend/app/schemas.py`): `DatasetReadiness`, `DatasetReadinessCheck`,
  `ReadinessState`, `ReadinessAction`, `DatasetPrepStatus`, `PrepStep`, `DatasetDetection`,
  `DatasetFieldMapping`, `DatasetPrepPlan`, `PrepDecision`, `PrepEngine`,
  `DatasetPrepStartRequest`, `DatasetPrepApplyRequest`, `DatasetPrepJobRead`, `PrepTransform`,
  `DatasetTablePage`, `DatasetTableColumn`, `DatasetTableRow`, `DatasetCellKind`.
  `DatasetSummary` gains `readiness` and `prep`; `DatasetPrepPlan` gains `transform`;
  `DatasetReadiness` gains `busy`; `DatasetPrepStatus` gains `step`, `detail`, `progress`.
- Frontend surfaces: `frontend/features/datasets/prep/` (overview tab, plan review, readiness
  panel, ingest drop card, progress readout, split preset control, transform panel, hooks),
  `readiness-badge.tsx`, `new-dataset-panel.tsx`, `data-grid.tsx`, a rewritten `detail-tabs.tsx`,
  and a readiness-aware dataset select in `training-page.tsx`.
- Storage/DB changes: `storage/datasets/<id>/_staging/` (raw files, relative paths preserved),
  `storage/datasets/<id>/_derived/prepared.jsonl` (transform output, rewritten per run),
  `storage/datasets/<id>/prep_undo.json` (pre-apply snapshot), `metadata.prep` in the dataset
  manifest, and a new `data_prep_jobs` table with an Alembic revision.

## Behavior

### Readiness (Stage B)

`readiness_for()` in `backend/app/services/datasets/prep/readiness.py` is a pure function of
`task_type`, `format`, `labels`, `splits`, and `metadata`. It imports only from `app.schemas` —
`datasets/service.py` imports it, so importing back into `datasets/` would be a cycle.

It must stay pure over `DatasetSummary` fields because `summary()` runs for every dataset on every
catalog list, and `_split_summary` already walks items and reads annotation JSON per item. A second
filesystem walk would double that cost. Checks needing file contents are therefore split into two
tiers:

- **Structural** — on `DatasetSummary.readiness`. Free, always present, drives the training page.
- **Quality** — from `GET /datasets/{id}/eda`, fetched only on the studio detail screen, rendered
  as advisories.

The split is forced rather than stylistic: `_split_summary` sets `annotation_count = len(items)`
unconditionally for `llm_finetune`, so "every record has a non-empty output" is not derivable from
a summary. It comes from `_llm_eda_summary`, whose `empty_output_count` already lands in
`DatasetEdaSummary.unlabeled_count`.

Universal checks:

| id | severity | rule |
|---|---|---|
| `prep_idle` | blocking | `metadata.prep.state == applying`, or analysis over a dataset with no items → `blocked` / `wait` |
| `prep_applied` | blocking when the data rules fail, advisory otherwise | `metadata.prep.state` in `{draft, planned, failed}` |
| `train_non_empty` | blocking | `splits.train.item_count > 0` |
| `valid_non_empty` | blocking for image tasks, advisory otherwise | `training_root` raises `FileNotFoundError` without `valid/images` |
| `inbox_drained` | advisory | `splits.unassigned.item_count == 0` |

Per-task checks:

| task_type | blocking | quality tier |
|---|---|---|
| `classification` | ≥2 labels; train and valid `annotation_count > 0` | label coverage, imbalance |
| `object_detection` / `segmentation` | ≥1 label; train and valid `annotation_count > 0` | imbalance |
| `text_classification` | ≥2 labels; `train.annotation_count > 0` | label balance |
| `summarization` | `train.annotation_count > 0` (the annotation is the summary) | — |
| `question_answering` | `train.annotation_count > 0` (annotation carries question/answer) | — |
| `llm_finetune` | `format` in `LLM_FORMATS`; `train.item_count >= 10` | empty outputs, duplicates |
| `language_modeling` | `train.item_count > 0` | — |
| `tabular` | mapping carries `label_column`, ≥1 feature column, `target_task_type` | — |

The `llm_finetune` floor of 10 is `MIN_TRAIN_RECORDS` in
`backend/app/training/runners/llm_sft.py`, which raises below it. That runner also falls back to
the `unassigned` inbox when `train` is empty; readiness mirrors that tolerance rather than
contradicting it, reporting `needs_prep` rather than `blocked`.

States: `ready`, `needs_prep` (the agent can fix it), `needs_input` (a human must), `blocked` (a
job is running). `summary` is the one sentence the training page renders — the first failing
blocking check's `detail`.

**Prep state never downgrades data that already trains.** The first cut folded the whole run
lifecycle into the verdict — `{detecting, planning, applying}` blocked, and `{draft, planned,
failed}` forced `needs_prep` — which produced a real bug: asking Orinth to re-read an
already-trainable dataset dropped it to "Preparing…", and if that run stopped short of applying
(needing labels, say) it stayed at "Needs prep", none of which was true of a single file on disk.
Two rules fix it, and both follow from the same principle — whether a dataset can be trained on is
a property of its files, not of a job that ran over them:

- Only `applying` rewrites the splits, so only `applying` blocks. `detecting` and `planning` read
  `_staging/` and nothing else; they set `DatasetReadiness.busy` and leave the verdict alone. The
  exception is a dataset with nothing in its splits, where there is no prior verdict to preserve
  and the run in flight is the only honest thing to report.
- The data rules are evaluated **first**, and `prep_applied` is blocking only when one of them
  already failed. Then it carries the more specific message (the plan's `needs_input`, or "run
  Prep"); otherwise it stays in the check list as an advisory saying the run stopped early and the
  dataset is unchanged.

### Progress

`DatasetPrepStatus` carries `step` (a `PrepStep`), `detail` (one sentence), and `progress` (0..1)
alongside `state`. They are separate fields from `state` on purpose: `state` answers "can this be
trained on yet" and is read by readiness, while `step` answers "what is happening right now" and is
read only by the progress readout. Every stage transition in `run()` writes all three through
`_step()`, which swallows its own errors — a run that completed but failed to announce itself is
strictly better than one that died announcing itself.

`detail` is written server-side because only the server knows the counts ("Read 312 files — looks
like classification, checking"). The client polls `/prep/status`, which is a single manifest read;
polling `GET /datasets/{id}` instead would walk every split on every tick, which is seconds on a
large dataset and lands on the same thread pool the run is using.

### Browsing prepared data

`GET /datasets/{id}/table` returns the dataset as columns and rows — one shape for every modality,
so the studio has one grid instead of a thumbnail wall, a text preview list, and a records table.
`services/datasets/table.py` is a pure projection of the `DatasetItemSummary` list that
`list_items_page` already produces; it reads no files of its own, and `table_page` resolves the task
type from `_location` rather than `summary()`, which would walk every split on every page turn.

Deliberately **not** a materialized index (a per-split Parquet or SQLite table rebuilt on write).
An index would buy whole-dataset sort and filter, but the training runners read the on-disk layout
directly, so a second source of truth can disagree with it and the failure mode is silent — the
grid shows a row the trainer skipped. Paging does not need it: `list_items_page` decides the page
from the file paths and opens only that window. Whole-dataset sort and filter is what is deferred,
and when a dataset is large enough for it to matter the index belongs beside the splits as a
derived artifact with an explicit rebuild, not smuggled in under a browse endpoint.

Only cells with a real write route behind them are `editable`: the label
(`PATCH …/items/{split}/{id}/label`) and the split (`POST …/items/move`). `editable` is decided on
the server beside the column definition rather than guessed in the client, because a cell that
looks editable and silently discards the edit is worse than a read-only one. Detection and
segmentation get a class list and a region count rather than a single `label` cell — one label
would be a lie about which of many boxes it names — and editing those stays the annotation
editor's job.

### Ingest (Stage C)

`_save_uploaded_image` flattens every upload to `<split>/images/<stem>-<hex><suffix>`. That
destroys the directory structure which *is* the detection signal: `data.yaml` beside `train/labels/`
means YOLO detection, and sibling `normal/` and `kista/` folders mean classification with those
labels. Ingest therefore writes raw bytes to `storage/datasets/<id>/_staging/` preserving each
file's relative path, mirroring the tabular phase's never-mutated `source/` convention.
`_locations()` globs `*/manifest.json`, so `_staging/` stays invisible to the catalog.

The client sends a `relative_paths` list parallel to `files`. `webkitdirectory` exposes
`File.webkitRelativePath`; the `showDirectoryPicker` walk yields the path from its recursion. Each
segment is sanitized — `..`, absolute paths, and symlinks are rejected — and a missing entry falls
back to the upload filename.

A draft dataset carries a provisional `task_type` of `classification` so the `TaskType` literal
stays unchanged and nothing ripples into training filters, `_media_dir`, or project gating. Draft
state lives in `metadata.prep.state`, not in `task_type`.

`_require_project_task` is **not** called at ingest — nothing has been claimed yet. It is called at
apply, when the task becomes real, following the precedent the tabular spec set for
`mapping.target_task_type`. The planner also receives the project's `task_types` and never proposes
a task the project would reject.

### Detect and plan (Stage D)

Four sniffers are lifted from `DatasetService` mixin methods to module-level pure functions so
`detect.py` need not construct a service; each mixin method becomes a one-line delegate, leaving
every existing call site untouched: `_is_yolo_root`, `_labels_from_yolo_yaml`, `_labels_from_coco`,
and `_text_upload_rows` in `format_io.py`, plus `_detect_format` in `dataset_hub.py`, which moves
to `prep/detect.py::detect_record_format` and leaves one alpaca/sharegpt/messages/qa sniffer in the
codebase instead of two.

Detection evaluates top to bottom, first match wins, and reports a confidence plus a list of
human-readable `signals` that become the evidence shown in the UI.

Two rules run **before** every structural guess, mirroring the precedence `format_io._annotations`
already uses (JSON sidecar → YOLO → COCO), because a declaration beats an inference:

- **`manifest.json`** — a dataset Orinth exported, re-uploaded whole. Trusted outright.
- **`<split>/annotations/*.json` sidecars** — the `kind` field states the task. Both spellings are
  accepted: `DatasetAnnotation.kind` uses the short form (`summary`, `qa`) while the reference
  datasets on disk carry the long one (`summarization`, `question_answering`).

Both were added after running detection over the 14 real datasets in this repo, and both fix
misdetections found there. `storage/datasets/trash-clasification-*` is image classification whose
manifest names six classes, but it stores `<split>/images` beside `<split>/labels` — structurally
identical to a YOLO detection root — so it detected as `object_detection` with zero labels.
`datasets/clinical_notes_classification` and `datasets/medical_summarization` are stored in Orinth's
own `<split>/texts` + `<split>/annotations` layout and read as an unlabelled corpus.

A third rule unwraps **wrapper folders**: up to three levels of a single subdirectory with no files
beside it, stopping at any structural name (`train`, `images`, `annotations`, …) so a dataset that
holds only `train/` is not mistaken for a wrapper. People upload `my-dataset/` containing the
dataset, and archive tools add a level of their own — `sample_data/vision` is exactly this shape and
read as 300 unlabelled images while the dataset one level down declared six classes. Column vocabularies for the
delimited-file rows reuse those already in `_annotation_from_text_row` and `_text_from_upload_row`
rather than introducing a second vocabulary.

`heuristic_plan()` is the base case; `analyze()` decorates it. The heuristic path ships first and
always runs, so a plan exists with no key, no network, and no model. Defaults: preprocessing off
for image and record data, `nlp_clean` on for text, never `augmentation_mode="materialize"` (it
multiplies storage and nobody asked for it), and `train .7 / valid .2 / test .1, seed 42` matching
`defaultSplitConfig()`, unstratified for `llm_finetune`.

With a key, `analyze()` calls the model with a strict JSON schema and one repair turn. Auth,
rate-limit, transport, and unparseable-response failures each fall back to the heuristic plan with
a notice naming the reason. The parser drops every field naming something absent from the
detection: a task outside the project's allowed types, a label not in `candidate_labels`, a mapping
column not in `columns`, a transform outside `ALLOWED_PREPROCESS_TRANSFORMS`. **The agent never
invents a class taxonomy.** And `format` is never taken from the model when detection confidence is
at or above 0.85 — a YOLO root is a YOLO root. The model ranks and names; the deterministic stage
decides structure.

### Transform (Stage D2)

Detection names a task by matching structure against rules. That covers the shapes the ML world has
standardized on and stops dead at the shape most real data arrives in: a **feature table**. Kaggle's
early-stage diabetes set is sixteen clinical yes/no columns and a `class` column; a
student-performance export is study hours, attendance, and a grade. No column vocabulary maps those
onto a trainable task, because the mapping is not in the names — it is in what you decide to predict
from what. Both landed on "Orinth could not map these columns to a task it can train."

So the stage stops pattern-matching and generates a program. It runs **only when planning produced
no task**, so nothing above it changes.

- **`prep/sandbox.py`** runs a `transform(rows) -> rows` in a subprocess. The contract is the first
  boundary: the script receives a JSON list and returns one, and never learns where the dataset
  lives. Inside, an import allowlist enforced by a `sys.meta_path` hook — preceded by a purge of the
  dangerous modules CPython pre-imports, since `import x` consults `sys.modules` before `meta_path`
  and the hook alone would not stop `import os`. No `os`, no `socket`, no `subprocess`, and so no
  third-party dataframe library either, which would readmit all three transitively. Plus a
  `builtins.open` gate scoped to the work directory, CPU/file-size/descriptor ceilings (`RLIMIT_AS`
  on Linux only — macOS reserves enough address space at startup that a useful ceiling kills the
  interpreter first), a scrubbed environment carrying none of the platform's keys, and a wall clock
  enforced over a new session so a child that forks anyway dies with its process group.

  The threat model is stated in the module and is deliberately modest: the code is written by a
  model this codebase prompted, and the realistic failure is *careless* — a stray `requests.get`, an
  unbounded loop, a write into the dataset. A determined escape from in-process CPython is not
  stoppable this way, and if untrusted third-party code ever runs here it needs a container.

- **`prep/transform.py`** has two engines and one contract. `builtin` picks a target and feature
  columns from column statistics and emits a template with those choices baked in as a literal
  config. `llm` sends the column profile to OpenRouter and takes back a script. Both produce a
  script, and both scripts run in the same sandbox — the builtin could have been an in-process
  function, and that would have been faster and would have grown a second implementation that drifts
  from the one users can read. One path, one artifact: what the Prepare tab shows is what ran.

**Choosing the target** is the decision that matters most, and "the column with the fewest classes"
was the first rule tried and is wrong in a way that looks right — on a student export it picks
`gender` over `result`, produces a perfectly valid dataset, and trains a model for a question nobody
asked. Two conventions beat it: a column *named* after an outcome (`class`, `result`, `exam_score`,
rightmost match winning, so `exam_score` beats `previous_scores`), then the *last* column, which is
where essentially every UCI and Kaggle table puts the target. Only then does shape decide, and there
the demographic attributes are tried last. A chosen column that is continuous is cut into thirds,
which changes the question from regression to three-class classification and is therefore stated in
the plan rather than left to be inferred from a suspiciously round accuracy.

**Serialization is `column_value`, not `column: value`.** The NLP preprocessing preset strips
punctuation with `[^\w\s]`, which would sever every column from its value and leave a bag of bare
`yes`/`no` tokens with the pairing — the entire signal — destroyed. Underscores survive that regex.
Numeric columns with more than a dozen distinct values are bucketed into quartiles: one token per
distinct number carries nothing a bag-of-words classifier can use.

**Output is validated against the claimed task before anything is written**, on the same principle
as `plan.py`: a model that invents a taxonomy is the damaging failure. Rows must carry the task's
required keys; `text_classification` must yield at least two and at most 100 distinct labels; and at
least half the input rows must survive, because a transform that returns a hundred rows out of five
thousand has not been selective, it has been wrong. Any failure — bad JSON, a blocked import, a
timeout, a single label — falls back to the builtin with the reason stated on `PrepTransform.notice`.

**A dominant feature is reported as a fact, not a verdict.** When one column closes 75% of the gap
between guessing the commonest class and being right every time, the rationale says so: `exam_score`
"predicts" a `Pass`/`Fail` column that was computed from it. Lift rather than raw accuracy, because
raw accuracy is dominated by class balance — on a 79%-Pass table every feature scores 0.79 by always
answering `Pass` and the standout scores 0.96, two numbers that look similar and mean nothing alike.

This began as a leakage warning that told the user to drop the column, and measurement killed that.
At the quartile resolution the transform emits, a genuinely derived column scores 0.80 and iris
petal width 0.77; at ten buckets, 0.96 and 0.93; at twenty, iris petal *length* reaches a flat 1.00.
Finer buckets do not separate leakage from a strong feature — they shrink every group until all of
them are pure. **Nothing computable from the table alone makes that distinction**, so the note
reports the measurement and asks the reader whether the column was computed from the target.
Advising a drop would, on the average table, delete the most useful column in it.

Output goes to `_derived/prepared.jsonl`, a sibling of `_staging/` and invisible to the catalog for
the same reason. Apply reads `_derived/` **in place of** the raw row files when it exists — importing
both would ingest every row twice — and images and `.txt` files pass through untouched. Undo and
"discard raw files" both delete it: leaving it would make a later apply, planned against the raw
upload, silently read the reshaped rows instead.

### Transparency

Every plan field carries a `PrepDecision` with `source` (`detected` / `heuristic` / `llm` / `user`),
`confidence`, a one-sentence `rationale`, and `evidence` drawn from `detection.signals`. Evidence is
what makes a decision checkable rather than merely asserted: the user verifies "312 files across 3
folders: normal, kista, granuloma" against what they uploaded without having to trust the agent.

`PrepEngine` records the mode, the model, a `notice` explaining any degradation, and token and cost
counts. Heuristic output is never presented as model output.

### Apply and reversibility (Stage E)

Apply runs entirely over existing service methods: `_update_manifest` and `_create_layout` for the
real task/format/labels, per-modality writers to move staged files into `unassigned/`,
`update_dataset` for the preprocess config, and `process_dataset` for the split — unchanged.

Apply also *rebuilds* rather than appends: every item under `<split>/` came from a previous apply
over the same staging directory, so the split directories are dropped first. Without that, a second
run — after an Undo, or after editing the plan — imports the upload on top of itself and silently
doubles the dataset. The guard is the presence of `_staging/`: a dataset the agent staged is one the
agent can rebuild from source, which is not true of a hand-built one.

`_staging/` is retained after apply, so re-running prep with a corrected plan needs no re-upload.
That retention is what makes auto-apply safe rather than reckless. A pre-apply snapshot in
`prep_undo.json` backs the Undo action, which restores rather than replaying inverse operations. A
"Discard raw files" action under Advanced reclaims the space.

Auto-apply is the default and is a flag on the start request, so the review-gate variant stays one
boolean away: the non-auto terminal state is `planned`, and `/prep/apply` runs the same code either
way.

### Jobs

Datasets stay filesystem-native, but the prep *job* gets a `data_prep_jobs` row. Every other job in
the app has one, and it is what supplies progress polling, `list_jobs`/`delete_jobs`, and
`reconcile_stale_jobs` from `job_runner.py` for free, plus real columns for token and cost
accounting. This departs from phase 11's filesystem-only choice deliberately: recipes are a working
area, whereas a prep run spends money and must survive a restart. The manifest also carries
`metadata.prep`, so a dataset stays self-describing without the DB — which is what lets readiness
work on reference and sample datasets that have no job row.

### Frontend

The catalog leads with a single `NewDatasetPanel`. It replaced four peer affordances — an ingest
drop card plus "Add Dataset", "Browse HuggingFace", and "New from documents" buttons on the toolbar
— which was four decisions before a byte moved, three of them the wrong default. The panel asks one
question (where is the data) through the existing `.segmented-control`: **Files** (the drop card,
default), **HuggingFace** (`HubImportPanel`), **Documents** (a handoff to `/datasets/recipes`, which
needs a model and generation choices and so keeps its own screen), and **Empty**
(`DatasetCreatePanel`). Nothing was removed; three things were demoted. `showCreate`/`showHub` are
gone from `DatasetPage` along with `setShowCreate` from `useCreateDatasetMutation`: the panel owns
which source is showing, so there is no longer a pair of booleans kept mutually exclusive by hand.

While a run is in flight, `PrepProgress` renders the stage ladder — read the files, work out the
task, reshape the rows, build the splits — with the current rung lit and the server's own sentence
under it ("Read 312 files — looks like classification, checking"). The sentence comes from
`prep.detail` rather than being mapped from the step client-side, because only the server knows the
counts, and a count that moves is what separates progress from a hang. `transforming` is drawn only
once reached, since a rung nobody will visit reads as a stall. It appears in two places from one
component: under the drop card while uploading, and on the Overview tab afterwards, so navigating
into the dataset moves the readout instead of restarting the explanation.

The Overview tab no longer offers "Prepare with Orinth" as the primary action on a dataset that
already trains. Primary becomes **View data**; re-running is a secondary **Re-run Orinth**. Offering
the agent as the headline action on finished data is what made a redo look like the expected next
step.

The studio's four peer tabs become `Overview | Data | Prepare` with a completion dot per tab driven
by `readiness.checks`. Data itself carries a **Table / Gallery** switch, defaulting to Table: a grid
is the only view that works for every modality and the only one that answers "what is in here"
without scrolling, while the gallery and records views remain where the annotation editor and the
record drawer live. Full replacement is safe because Overview and Prepare *compose* the existing
`PreprocessPanel`, `SplitConfigPanel`, and `VersionPanel` rather than reimplementing them.

The primary action, "Make ready to train", wraps `process_dataset` **only**. `create_version` is
not chained into it: `prepared_training_root` resolves the live dataset root and never reads
`storage/dataset_versions`, and no training, evaluation, or export path reads a version artifact.
Presenting it as a step toward training was the source of the Proceed-vs-Create-version confusion.
It moves under Advanced as "Snapshot (archive)" with copy that says what it is.

Split ratios become a preset row with a Custom option; the three raw inputs remain under Custom
alongside a live normalized readout, because `_normalize_split_config` already divides by the total
and only the UI ever implied otherwise.

### Design

Per `frontend/DESIGN.md`: no new tokens, no shell change, no new route — the studio stays at
`/datasets?dataset=`. Readiness badge tones pair `-tint` background with `-strong` text (`ok`
ready, `warn` needs_prep, `danger` needs_input, `info` blocked), since §9 requires 4.5:1 at the
11px badges render at. The plan source badge uses `info` for `llm` and `neutral` for `heuristic`,
matching the phase-11 record-provenance precedent. The prep status strip reuses `.recipe-stepper`;
the Advanced disclosure reuses the phase-13 `.advanced-panel` accordion.

## Data Flow

```
files[] + relative_paths[]
   │
   ▼
storage/datasets/<id>/_staging/**      (raw, structure preserved, never mutated)
   │
   ├── detect()   deterministic  ──►  DatasetDetection {task, format, labels, signals, confidence}
   │
   ├── profile()  eda_summary()  ──►  DatasetEdaSummary
   │
   ├── plan()     heuristic, then LLM refinement if a key is configured
   │                             ──►  DatasetPrepPlan + [PrepDecision] + PrepEngine
   │
   ├── transform()  only when plan.task_type is still None
   │      model-written or template Python → prep/sandbox.py (no net, no fs, rlimits)
   │      → validated against the claimed task
   │                             ──►  _derived/prepared.jsonl + PrepTransform
   │
   └── apply()    _reset_items → _create_layout → _update_manifest
                  → move staged files (or _derived/ rows, when a transform ran)
                  → update_dataset(preprocess) → process_dataset(split)
                             │
                             ├──►  prep_undo.json      (pre-apply snapshot)
                             └──►  manifest.metadata.prep = {state: ready, plan, decisions, engine}
                                            │
                                            ▼
                                   DatasetSummary.readiness  ──►  catalog card
                                                                  studio header
                                                                  training select
```

## Edge Cases

- **No OpenRouter key** — heuristics produce the full plan; the notice names the missing key.
- **Key rejected, rate limited, or network down** — fall back to the heuristic plan with the reason
  stated; the run still completes.
- **Model returns unparseable JSON** — one repair turn, then the heuristic plan.
- **Model names a label or column that is not in the data** — dropped by the parser.
- **Flat unlabeled images** — no taxonomy is invented; the dataset lands `needs_input` and readiness
  names what is missing.
- **Fewer than 10 LLM records** — `needs_prep`, not `ready`; the runner would raise.
- **Legacy dataset with no `metadata.prep`** — readiness computes normally; the catalog renders.
- **Reference and shared sample datasets** — readiness computes with no job row.
- **A task the project has not declared** — the planner never proposes it; apply returns 409.
- **Upload with no `relative_paths`** — falls back to filenames; detection degrades to the flat-file
  rows rather than failing.
- **Path traversal in a relative path** — rejected per segment.
- **Prep job orphaned by a restart** — `reconcile_stale_jobs` flips it to `failed` on boot.
- **Re-running prep after apply** — staged files are still present, so it re-plans without re-upload,
  and the split directories are rebuilt so the dataset is not doubled.
- **A feature table with no mappable columns** — the transform stage generates one; with no key it is
  the deterministic template, and the plan says which engine wrote it.
- **Generated code that will not run** — a syntax error, a blocked import, a timeout, or a non-JSON
  return all fall back to the builtin transform with the reason on `PrepTransform.notice`.
- **Generated code that runs but is wrong** — a single label, more than 100 labels, or under half the
  rows surviving is rejected the same way.
- **A table whose only closed-set column is `gender`** — demographic attributes are chosen last, and
  only when nothing is named like an outcome and the last column cannot serve.
- **A continuous target** — bucketed into thirds, with the cut points and the regression-to-
  classification change stated in the plan.
- **One feature that all but determines the target** — reported with the measurement and the
  question; never dropped, because leakage and a strong predictor are indistinguishable from here.
- **A project that cannot train text** — no transform is proposed and the dataset stays `needs_input`.
- **Undo after a transform** — `_derived/` is deleted with the manifest restore, so a later apply
  reads the raw upload it was planned against.

## Acceptance Criteria

- Behavior:
  - Dropping an unmodified YOLO folder, a per-class image folder, an alpaca JSONL, or a labelled CSV
    yields a `ready` dataset with correct task, format, labels, and populated splits.
  - Flat unlabeled images yield `needs_input` with a reason, and no invented labels.
  - A feature table with no mappable columns yields a `ready` dataset via a generated transform,
    with the target column, the row counts, and the script itself shown in the Prepare tab.
  - Generated code cannot open a socket, import `os`, read a file, or run past its wall clock.
  - Every agent decision displays a source, a rationale, and its evidence.
  - Undo restores the pre-agent state.
  - A dataset that is not ready is still selectable on the training page and states why.
  - No training run reads `storage/dataset_versions`.
- Tests: see Validation.
- Manual checks: the six-row modality table and the transparency checks in Validation.

## Deferred

- Readiness-aware selection on the testing and inference pages, which repeat the same four-screen
  task-type re-selection problem.
- `.zip` ingest, which needs an extraction and zip-bomb guard.
- OCR for scanned documents.
- Retiring `services/recipes/openrouter.py` in favour of `services/providers/`. Phase 11 is shipped
  and this phase does not refactor it; the duplication is recorded here deliberately.
- Removing `create_version` outright. This phase demotes it to an archive action rather than
  deleting an endpoint that may have out-of-band users.

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd backend && uv run alembic upgrade head && uv run alembic downgrade -1 && uv run alembic upgrade head
cd frontend && pnpm generate:api
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm test
cd frontend && pnpm build
```

New tests:

- `backend/tests/test_dataset_readiness.py` — one case per rule-table row from `DatasetSummary`
  fixtures, no filesystem; plus a legacy manifest with no `metadata.prep`.
- `backend/tests/test_dataset_prep_detect.py` — one synthetic tree per detection-table row.
- `backend/tests/test_dataset_prep_plan.py` — heuristic determinism; the parser drops an invented
  label, an unknown column, a disallowed transform, and a task outside the project's types; format
  override rejected at high confidence; repair retry; auth, rate-limit, and transport failures each
  fall back with the right notice, using an injected mock transport.
- `backend/tests/test_dataset_prep_service.py` — ingest → prep → apply → splits populated and
  `readiness.trainable`; undo restores; stale-job reconciliation; 409 on a disallowed task.
- `backend/tests/test_dataset_prep_sandbox.py` — every allowlisted module imports; `socket`,
  `urllib`, `http`, `ssl`, `os`, `subprocess`, `shutil`, `pathlib`, and `ctypes` do not; a file
  outside the work directory cannot be opened; a runaway transform is stopped; a missing entry point,
  a raised exception, a non-list return, and an over-cap row count each report rather than crash; the
  server's `OPENROUTER_API_KEY` is not reachable from inside.
- `backend/tests/test_dataset_prep_transform.py` — detection alone cannot place a feature table (the
  precondition, and the test that fails first if this stage becomes dead code); the diabetes and
  student column shapes both reach `ready`; the target comes from the name and not the class count;
  a continuous target is bucketed and says so; identifier columns are never features; a column that
  dominates the target is reported; items carry `column_value` tokens; staging is byte-identical after a
  run; undo and discard both clear `_derived/`; a re-run does not double the dataset; a project
  without the task gets no transform; a good model script is used and six kinds of bad one each fall
  back with a specific notice.
- `frontend/features/datasets/readiness-badge.test.tsx`, `prep/prep-hooks.test.tsx`, and an updated
  `detail-tabs.test.tsx`.

Manual verification:

| # | Input | Expect |
|---|---|---|
| 1 | unmodified YOLO folder | `yolo` / `object_detection`, labels from `data.yaml`, reaches `ready` |
| 2 | images in per-class subfolders | `image_folder` / `classification`, labels from folder names |
| 3 | alpaca `.jsonl` | `instruction_jsonl` / `llm_finetune`, unstratified; under 10 records lands `needs_prep` |
| 4 | CSV with text and label columns | `text_classification`, mapping shown in the plan review |
| 5 | flat unlabeled images | `needs_input`, missing labels named, no invented taxonomy |
| 6 | legacy dataset with no `metadata.prep` | readiness computes, catalog renders, nothing 500s |
| 7 | feature table (iris, diabetes risk, student performance) | transform generated, target named, `ready` |

Transparency checks: with no key every decision reads `heuristic` with a notice naming the missing
key; with a key decisions read `llm` with rationale and evidence and the engine reports model and
cost; a forced 401 degrades with a stated reason rather than failing. The OpenRouter key must appear
in no API response, log line, job row, or manifest.

End to end: upload, go straight to `/training` without visiting Prepare, confirm the dataset is
pre-selected with a `ready` badge and an enabled Start button, then train from it and confirm
`prepared_training_root` still resolves.

### Live verification performed

Against a running `uvicorn` + `next dev` pair, over the real `storage/` tree:

- `GET /api/datasets` returns readiness for all 12 datasets on disk (4.5s — pre-existing cost of
  `_split_summary`, which readiness does not add to since it is pure over the counts already
  gathered).
- `POST /api/datasets/ingest` with four files and their `relative_paths` created a draft reading
  `needs_prep`; `POST /api/datasets/{id}/prep` returned `classification` / `image_folder` with
  labels `[NORMAL, PNEUMONIA]`, splits 2/2/0, and readiness `ready`.
- That run used a **real OpenRouter call** (`openai/gpt-4o-mini`, 386 prompt + 70 completion
  tokens, $0.0001 recorded on `PrepEngine`). Every decision still reported `source: detected` —
  detection was above the structure-lock threshold, so the model contributed only the rationale,
  which is the intended division.
- `/datasets`, `/datasets?dataset=…`, and `/training` all render without client errors, and the new
  surfaces are present in the served route bundle.
- The test dataset was deleted afterwards; `storage/datasets/` is unchanged.
