# Spec: Phase 17 Model Architecture Studio

## Status

**Implemented — all five stages, plus a second emitter.** Compose a graph (45 node types, 20 starter
templates), validate it, save it, read and download the generated Python as **either TensorFlow or
PyTorch**, and train it: image and text classifiers into the model registry, and from-scratch
transformers on next-token prediction with sample generation.

### Revision 2: real model architectures, not just parameter counts

Configs verified against published sources in August 2026 (HuggingFace
`transformers` model docs and the DeepSeek-V3 report), recorded per preset in
`LlmPreset.source`. The features below are implemented in **both** emitters,
so a preset's generated code has the shape of the real model rather than a
generic transformer carrying its parameter count.

- **QK-norm** — RMSNorm over the query and key head dimensions before the dot
  product. Qwen3's signature; Gemma 3+ uses it too.
- **Sliding-window attention with a hybrid layer pattern** — `sliding_window`
  plus `global_every`, so most layers see a local band and every Nth sees the
  whole sequence. Gemma 4 runs 512-token windows at 5 local to 1 global.
- **Tied embeddings** — the LM head projects with the transpose of the
  embedding table instead of learning a second matrix. Materially changes the
  count: at Gemma 4's 262k vocabulary and 2304 width it removes 604M
  parameters.
- **Logit softcapping** — `tanh(logit / cap) * cap`, Gemma's
  `final_logit_softcapping`.
- **Shared experts** — always-on unrouted experts alongside the routed ones.
  DeepSeek V3 pairs 1 shared with 256 routed.
- **Decoupled head dimension** — Gemma's 8 heads of 256 against a 2304-wide
  residual stream. Deriving `head_dim` as width/heads would give 288 and build
  a different model, so presets state it.

**New named presets, with the counts a test asserts:** Qwen3 0.6B (596M),
Qwen3 8B (8.19B), Qwen3 30B-A3B (30.55B — the published figure exactly),
Gemma 4 sparse and dense, and a DeepSeek-style shared-expert MoE.

**Two presets deliberately do not match their namesake's headline size, and
say so in their own descriptions.** Gemma 4's published 26B-A4B adds per-layer
embeddings (PLE) and an expert count its public config does not state; the
preset is built from the documented `Gemma4TextConfig` defaults and lands near
16B. DeepSeek V3's multi-head latent attention has no node on this canvas, so
its attention is modelled as grouped-query and the total lands above 671B.
Naming them after the checkpoints they approximate would have been the easy
lie; the descriptions name the gap instead.

### UI revision 2

- **Select / Move tools.** Drag-on-canvas either marquee-selects or pans,
  rather than requiring a modifier for whichever one is not the default.
- **Group frames** — titled, resizable, colour-pickable annotation rectangles
  drawn behind the nodes. They live in `training_defaults.groups`, never in
  `nodes`, so no backend validator, shape rule, or emitter has to learn to skip
  them. Tints come from `labelFill` at 9%, the sanctioned translucent helper.
- **Branch-aware edges.** An edge leaving a fan-out or entering a merge is
  drawn as a bezier; a plain link in a chain stays a step. A step edge at a
  branch reads as one path, which is the opposite of the truth exactly where a
  residual or a router needs to be legible.

### Revision: scale, sparsity, and a second framework

- **A PyTorch emitter** (`emit_torch.py`) renders the same resolved graph as an `nn.Module`. Keras
  stays the compile target that trains in-platform; torch is an export. Both read one IR and one
  shape pass, so they cannot describe different models — a `slow` test asserts they agree on a
  transformer's parameter count exactly. Two nodes are untranslatable and say so rather than
  emitting something plausible: a `tf.keras.applications` backbone, and a custom layer written as a
  Keras `Layer`. The torch module is NCHW while the canvas shows NHWC, which the generated header
  states.
- **Grouped-query attention and mixture-of-experts nodes**, plus `num_kv_heads` / `ffn: "moe"` on the
  composite block. Both are emitted as generated helper classes in each framework. MoE experts are
  gated (SwiGLU), matching Mixtral — an earlier draft emitted 2-matrix GELU experts while the
  estimate assumed 3, which the torch build caught as a 50M-parameter disagreement.
- **Nine LLM scale presets, 6M to 70B**, in `llm_presets.py`. Each reproduces a published
  open-weight configuration, and a test asserts the parameter count matches: Mistral 7B →
  7,243,140,352; Llama 3 70B → 70,560,552,192; Mixtral 8×7B → 46,711,541,248. They are *blueprints* —
  the spec is explicit that nothing above ~200M trains on a workstation, and the local-build test
  suite skips them by name.
- **Catalog maximums were raised** (`layers` 64→256, heads 64→256, embedding width 8192→32768).
  A silent clamp had been turning the 70B preset into a 57B one; a test now asserts no preset is
  clamped.
- **`AdvancedParameterSpec` gained a `code` type**, and the studio renders it as a real editor —
  syntax-highlighted, with line numbers and static lint for the mistakes that actually happen when
  writing a Keras layer in a field (no `call()`, missing `super().__init__`, tab indentation,
  unbalanced brackets).
- **Pane sizes are user-controlled and persisted.** Palette, inspector, and the code/issues drawer
  each have a draggable, keyboard-operable separator; sizes live in `localStorage` per pane.
- **The class-count preview is stored on the architecture** (`training_defaults.num_classes_preview`)
  rather than reset to 2 every visit — the concrete complaint that a small CNN "always gives only 2
  classes" on the canvas.
- **Canvas interaction:** box selection and multi-select (delete acts on the whole selection),
  smoothstep edges so a branch reads as two paths rather than one, and a Tidy that fans siblings out
  symmetrically around their parent instead of stacking them downward.
- **Category icons and colours** from the `--label` data-visualization ramp — the sanctioned
  categorical exception in `DESIGN.md` §2, confined to the node stripe and palette dot.
- **List search and pagination** (12 per page, client-side over the graph-less summaries), and a
  search over the now-20 starter templates.

### Bugs fixed in this revision

- **`/api/datasets` was returning 500 for every caller.** A dataset written by the removed
  Parquet-backed `tabular` kind (commit b70baa3) failed `DatasetSummary` validation, and one stale
  manifest took the whole catalog down. Added `tabular → text_classification` and `table → csv` to
  the existing alias mechanism, and wired the alias validators onto `DatasetSummary` so a manifest
  that outlives its schema still lists.
- **`.visually-hidden` was never defined**, so the Import file input rendered as a raw
  `Choose File / No file chosen` control next to the styled button.
- **Two tests passed vacuously on a developer machine.** Both built `Settings` without stating the
  credentials they assert on, so the real `.env` leaked in — one asserted "no OpenRouter key" on a
  machine that has one. Both now state their own preconditions.

### Stage D and E deviations from the draft

- **`AdvancedParameterSpec` gained a `code` type**, rendered as a mono textarea. Custom layer bodies
  are the first param that cannot fit a single-line input, and adding a type kept the inspector on
  the one shared field renderer rather than forking it.
- **A new `Generation` advanced group** holds sampling temperature and length. It is registered in
  `app/ml/common/advanced.py`, the frontend `GROUP_ORDER`, and
  `test_advanced_training.py`'s documented set — that test caught the omission, which is what it
  exists for.
- **There is no generic Group/subgraph node.** Stacking is a `layers` param on a composite
  `transformer_block` node, which emits a Python loop rather than N copies of a subgraph. Generic
  nested subgraphs would have doubled the IR and emitter for one use case, and the palette also
  ships the individual primitives (RMSNorm, RoPE, MHA, SwiGLU) so a block can still be wired by hand
  and edited piece by piece — the `transformer_primitives` template does exactly that.
- **Keras ships no RMSNorm, RoPE, SwiGLU, or learned positional layer**, so the emitter generates
  them as helper classes above `build_model`, and only when the graph actually uses them. A plain
  CNN's generated file carries no transformer machinery.
- **A backbone-style causal mask node was dropped** in favour of a `causal` toggle on the attention
  and block nodes. Keras exposes it as `use_causal_mask=`; a separate node would have been a
  parameter pretending to be a layer.
- **Custom layers are hoisted once per class name.** Two nodes may share a class; emitting it twice
  would be a redefinition, so the first wins and later nodes reuse it.
- **Custom nodes may declare an output shape.** The analytic pass cannot read Python, so a blank
  declaration means "shape unchanged" (true of most custom layers, and it keeps downstream nodes
  resolvable) and anything else must be stated. Custom nodes always make the parameter estimate
  `None`.
- **`language_modeling` is a real `TaskType`**, resolving the spec's open question. It is grouped
  with NLP on the frontend (`isNlpTask` returns true, so the forms never offer it an image size)
  rather than with `llm_finetune`, which drives a different set of form branches entirely.
- **A separate `architecture_lm` family and runner.** The two graph runners share no dataset
  pipeline — one windows a corpus into next-token pairs, the other batches labelled images — so
  splitting them beat threading a mode flag through one runner. They share the command builder.
- **The LM runner has its own tokenizer**, word or character level, built from the training corpus
  and capped at `vocab_size`. It writes `tokenizer.json` beside the weights, because the vocabulary
  is not recoverable from the weights and the model is unusable without it.
- **LM sample generations reuse phase 14's `sample_generations.json`**, so the training detail page
  renders them with no new frontend code.
- **A from-scratch LM has no inference path.** `model_registry.predictor_for` raises a specific,
  actionable message rather than a bare "unsupported family". Samples on the run's detail page are
  the way to see what it learned.

### Stage B and C deviations from the draft

- **`@xyflow/react` is styled from `base.css` only**, not its `style.css` theme. Every visible
  surface — nodes, handles, edges, controls, minimap — is restyled from platform tokens, so no
  vendor theme enters the app and `frontend/DESIGN.md` §5's border-first rule holds on the canvas.
- **Architectures live under a `Catalog | Architectures` segmented control on `/models`**, not a
  new sidebar entry. Architectures *are* models; a top-level nav item would imply otherwise, and
  the sidebar's `pathname.startsWith("/models")` already lights up for the new routes.
- **The node inspector is the phase-13 `AdvancedField`**, exported from
  `frontend/features/training/advanced-settings.tsx` for reuse. This is the payoff of reusing
  `AdvancedParameterSpec` for node params: a node type declared on the backend gets its settings UI
  with no frontend change.
- **A "Preview classes" control lives in the studio header.** `units_from_dataset` heads have no
  resolvable width until a dataset is chosen, so the canvas needs a stand-in to show real shapes and
  parameter counts before a training run exists.
- **Training is one catalog option, `architecture_graph`**, with the graph chosen via
  `hyperparameters.architecture_id` — no `TrainingJobCreate` field, exactly as phase 13 intended.
  The key is stripped from the advanced payload before the runner sees it, since it is routing
  rather than a hyperparameter, and both `create_job` and the command builder reject a missing or
  broken architecture before any subprocess starts.
- **`_command_for_job` gained an optional `db` parameter.** Only the architecture family needs to
  read a row to build its command; every other family builds purely from the job's own parameters,
  so the session stays optional rather than coupling them all to the database.
- **The runner takes no `--image-size`.** Resolution comes from the built model's input shape, so
  the graph's Input node genuinely owns preprocessing and a form value cannot silently contradict
  the canvas. `TrainingJobCreate.image_size` is still sent (the schema requires it) and ignored.
- **`keras_common.py` was extracted** from `keras_classification_train.py` and is shared by both
  runners: dataset loading, `image_datasets`, optimizers, LR schedules, callbacks, class weights,
  and `write_evaluation`. Behavior of the existing runner is unchanged; the module-level names it
  used to own are re-exported from it so nothing that imported them breaks.
- **A graph-built model registers with `family="keras_classification"`.** The artifact is an
  ordinary `.keras` file, so testing, inference, and export need no knowledge that a canvas produced
  it — the whole point of choosing Keras as the compile target. `generated_model.py` is copied
  beside the weights for provenance.
- **Import from `.keras` and from a registry model is still deferred** (drafted for stage B, moved
  to a later pass). `.json` import, export, auto-layout, and templates all ship.
- **`POST /{id}/compile` remains unimplemented.** With training working end to end, the subprocess
  build check is now largely redundant — the analytic pass catches structural errors and a real
  training run is the authoritative build. It stays in the plan but is no longer on the critical
  path.

### Stage A deviations from the draft

All made either to match a registry/type constraint the draft did not account for, or because a
feature belongs to a later stage:

- **The stage A catalog omits Group, CustomLayer, CustomFunction, and every transformer primitive.**
  Those are stages D and E. What ships is I/O, Core, Convolution, Normalization, Recurrent, Merge,
  Regularization, and Backbone — 31 node types. `ArchitectureGraph` correspondingly has no
  `custom_nodes` or `groups` field yet; adding them is additive and needs no migration.
- **`POST /{id}/compile` is not implemented.** Stage A ships the analytic `/validate` only. The
  compile subprocess lands with the training runner in stage C, which is where the subprocess
  plumbing it needs already exists. `ArchitectureCompileResult` is likewise deferred.
- **Import accepts `.json` only.** `.keras`/registry-model import needs a TensorFlow subprocess to
  read the model config, so it lands in stage B alongside the UI that surfaces it. There is no
  `importers.py` yet; the JSON path lives in the service.
- **Bidirectional is a param on LSTM/GRU, not a wrapper node.** A node that wraps another node has
  no natural representation on a flat canvas, and this is the only Keras wrapper the palette needs.
- **Shape-valued params ride as comma-separated text** (`"224,224,3"`), parsed by
  `graph.parse_shape`. `AdvancedParameterSpec` has no tuple type, and inventing one would have
  meant a new field renderer on the frontend — the whole point of reusing the type. For the same
  reason `kernel_size`, `strides`, `pool_size`, and `padding` are single ints: square kernels and
  symmetric padding only. Non-square kernels would need a tuple type.
- **`MaxPool2D`/`AvgPool2D` spell "match the pool size" as `strides: 0`**, since the spec type has
  no unset value for a number. The emitter maps 0 back to Keras's `None`.
- **Arity is only enforced on nodes that reach the Output node.** A node the user has dropped but
  not wired up yet is a warning, not an error — otherwise placing a node would block saving or
  training an otherwise-complete model.
- **A pretrained backbone is emitted without `name=`.** `tf.keras.applications` derives its weights
  download filename from the model's name, so naming a backbone after its canvas node sends Keras
  looking for `<node>_notop.h5` and the ImageNet download 403s. Found by building the generated
  code rather than by reading it. The backbone keeps its stock name in `model.summary()`.
- **`total_params_estimate` is `None` whenever any contributing node's count is unknown** rather
  than a partial sum. A total silently missing a backbone's several million parameters is worse
  than no total. For graphs without a backbone the estimate matches TensorFlow exactly, which the
  slow tests assert.
- **The graph builders `node`/`edge`/`chain` live in `ml/architecture/templates.py`** and are shared
  by the templates and the test suite, so a graph in a test is constructed exactly like a real one.
- **Endpoints added beyond the draft table:** `GET /node-categories` (palette order, so the
  frontend does not hardcode a second copy), `GET /{id}/code/download` (attachment response), and
  `POST /{id}/validate` (validate a saved graph without re-posting it).
- **`NodeSpec.task_types` filters the palette.** An empty list means universal. Filtering is a
  palette hint only: a saved graph is never rejected for using a node the current task type would
  not have offered.
- **Unrelated bug fixed in `app/core/database.py`.** `_stamp_or_upgrade` stamped a matching
  pre-Alembic database at the *baseline* revision and then upgraded, which replays every migration
  since baseline against tables that already exist. That was invisible while baseline *was* head;
  this phase's migration is the first one after the baseline, so it surfaced immediately. An empty
  metadata diff means the live schema equals current ORM metadata — which is head by definition —
  so the stamp now goes at `head`.

### Decisions locked before drafting

- **Compile target is the Keras functional API.** A graph becomes a `tf.keras.Model`, so the existing
  Keras training runner, `.keras` artifacts, `KerasClassificationPredictor`, `model_registry`, testing,
  inference, and export all apply to graph-built models with no parallel pipeline.
- **LLM support is layer-level.** The palette carries transformer primitives (embedding, RoPE, MHA,
  RMSNorm, SwiGLU, LM head) that compose into a full transformer built from scratch. This is a
  research/teaching surface at toy scale — a from-scratch model trained on local hardware will not be
  competitive with a pretrained base. `specs/phase-14-llm-finetuning.md` remains the path for real
  LLM work; this phase does not replace it.
- **Custom code nodes hold real Python**, materialized into the generated module and executed in the
  training/compile subprocess. See [Custom code and trust](#custom-code-and-trust).
- **Canvas is `@xyflow/react`** (React Flow), a new pnpm dependency. Nodes are plain React components
  styled from `frontend/DESIGN.md` tokens; no vendor theme is imported.

## Goal

A visual, node-based editor under the Models menu where a user assembles a deep-learning or transformer
architecture by dragging layer nodes onto a canvas and wiring them together, sees live output shapes and
parameter counts, and then trains, tests, runs inference on, saves, and exports the result through the
platform's existing model pipeline. The same graph exports to runnable Python and imports back from a
saved file, and any node the palette does not cover can be written as custom Python inline.

## Scope

In:

- A versioned graph IR (JSON) as the single source of truth for an architecture.
- A server-declared node catalog covering core, convolutional, normalization, recurrent, merge,
  attention/transformer, and pretrained-backbone nodes.
- Analytic validation and shape inference (no TensorFlow import), plus an authoritative subprocess
  compile check that builds the real model.
- Keras code generation: a standalone `generated_model.py` exposing `build_model(...)`.
- Canvas studio UI at `/models/architectures`, with palette, inspector, generated-code panel, and
  layer summary.
- Persistence in a new `architectures` table with per-save snapshots under `storage/`.
- Import from platform `.json`, from a Keras `model.to_json()` config, and from an existing registry
  model; export to `.json` and `.py`.
- Custom layer / custom function nodes holding Python.
- A new `architecture_graph` training family that trains a graph-built model and promotes it into the
  model registry like any other trained model.
- Group nodes with a `repeat` count, so a transformer block is authored once and stacked N times.

Out:

- A PyTorch emitter. The IR keeps emitters pluggable, but only Keras ships here.
- Distributed or multi-GPU training. Graph training uses the existing single-process runner path.
- Editing an architecture concurrently from two sessions. Last write wins, guarded by a version check.
- A full inference surface for from-scratch language models. Stage E ships training plus in-job sample
  generation; a `keras_lm` predictor on the `/inference` surface is deferred to a follow-up phase.
- Any change to phase 14 LLM fine-tuning. That path is untouched.
- Automatic hyperparameter search or architecture search.

## Interfaces

### API

New domain router `backend/app/api/routers/architectures.py`, registered in
`backend/app/api/routes.py` alongside the existing routers.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/architectures` | List, filtered by `project_id`, `task_type`. |
| `POST` | `/api/architectures` | Create blank, or from `template_id`. |
| `GET` | `/api/architectures/{id}` | Fetch graph + metadata. |
| `PUT` | `/api/architectures/{id}` | Save graph. Body carries `version`; a stale version returns 409. |
| `DELETE` | `/api/architectures/{id}` | Delete architecture and its snapshots. |
| `POST` | `/api/architectures/{id}/duplicate` | Copy as a new architecture. |
| `GET` | `/api/architectures/node-catalog` | Palette definitions, filtered by `task_type`. |
| `GET` | `/api/architectures/templates` | Starter graphs (small CNN, ResNet-ish block, BiLSTM classifier, decoder-only transformer). |
| `POST` | `/api/architectures/validate` | Analytic pass over a posted graph. Returns per-node shapes, param estimates, and issues. Stateless, so an unsaved canvas can validate. |
| `POST` | `/api/architectures/{id}/compile` | Subprocess TF build. Returns real shapes, exact param counts, `model.summary()` text, or the build traceback. |
| `GET` | `/api/architectures/{id}/code` | Generated Python as text. |
| `GET` | `/api/architectures/{id}/export` | `.json` download of the graph. |
| `POST` | `/api/architectures/import` | Multipart: `file` (`.json` or `.keras`) **or** `model_id`. Returns a new architecture. |

Training reuses the existing endpoints unchanged: `POST /api/training/jobs` with
`model_id: "architecture_graph"` and `hyperparameters.architecture_id`. No `TrainingJobCreate` change —
the dict was left open-ended for exactly this, per phase 13.

### Schemas

In `backend/app/schemas.py`:

- `ArchitectureNode` — `{id, type, label, position: {x, y}, params: dict, group_id: str | None}`
- `ArchitectureEdge` — `{id, source, source_port, target, target_port}`
- `ArchitectureCustomNode` — `{id, name, kind: "layer" | "function", code, params: list[AdvancedParameterSpec]}`
- `ArchitectureGraph` — `{schema_version, nodes, edges, custom_nodes, groups, training_defaults}`
- `Architecture` — `{id, project_id, name, description, task_type, framework, version, graph, created_at, updated_at}`
- `NodeSpec` — `{type, name, category, description, inputs, outputs, params: list[AdvancedParameterSpec], min_inputs, max_inputs}`
- `ArchitectureIssue` — `{severity: "error" | "warning", node_id: str | None, message}`
- `ArchitectureValidation` — `{ok, issues, node_shapes: dict[str, list], total_params_estimate, layer_count}`
- `ArchitectureCompileResult` — `{ok, summary, node_shapes, total_params, trainable_params, error}`

**Node params reuse `AdvancedParameterSpec` from phase 13.** That type already carries
`{key, label, type, default, min, max, step, options, help, group}` and the frontend already renders it
generically in `frontend/features/training/advanced-settings.tsx`. The node inspector is therefore the
existing field renderer pointed at a node's spec — a new node type gets its UI for free, and a param
cannot drift from what codegen consumes.

### Frontend surfaces

- `frontend/app/(platform)/models/architectures/page.tsx` — list, thin entrypoint.
- `frontend/app/(platform)/models/architectures/[architectureId]/page.tsx` — studio, thin entrypoint.
- `frontend/features/models/architectures/` — `architectures-page.tsx`, `studio-page.tsx`,
  `node-palette.tsx`, `node-inspector.tsx`, `canvas-nodes.tsx`, `code-panel.tsx`, `import-dialog.tsx`,
  `auto-layout.ts`, `graph-client.ts`.
- `frontend/lib/api/architectures.ts`, re-exported from `frontend/lib/api/index.ts` so
  `import { api } from "@/lib/api"` keeps working.
- The Models page header gains a segmented control, `Catalog | Architectures`. The sidebar keeps one
  Models entry; `pathname.startsWith("/models")` already covers the new routes.
- Platform selectors go in `frontend/app/styles/platform.css` under an `.arch-*` prefix.

### Backend modules

- `backend/app/ml/architecture/catalog.py` — node specs, grouped by category.
- `backend/app/ml/architecture/graph.py` — IR dataclasses, parse, topological sort, cycle detection.
- `backend/app/ml/architecture/shapes.py` — pure-Python analytic shape inference. No TF import.
- `backend/app/ml/architecture/emit_keras.py` — graph → Python source.
- `backend/app/ml/architecture/importers.py` — Keras config → IR, registry model → IR.
- `backend/app/ml/architecture/layout.py` — layered auto-layout for imported graphs.
- `backend/app/services/architectures.py` — CRUD, snapshots, compile orchestration; singleton wired in
  `backend/app/container.py`.
- `backend/app/training/runners/architecture_compile.py` — subprocess build check.
- `backend/app/training/runners/architecture_train.py` — training runner.
- `backend/app/training/runners/keras_common.py` — dataset loading, metrics, callbacks, and CSV
  normalization extracted from `keras_classification_train.py` and shared by both runners. The existing
  runner keeps its current behavior and CLI; this is a refactor, not a rewrite.

### Storage and DB

New table `architectures`:

```
id           String(64)  PK
project_id   String(64)  index, nullable
name         String(120)
description  Text        nullable
task_type    String(64)  index
framework    String(32)  default "keras"
version      Integer     default 1
graph        JSON
created_at   DateTime    index
updated_at   DateTime
```

Alembic migration generated with `uv run alembic revision --autogenerate`, reviewed, and committed with
the model change, per `docs/ai/rules.md`.

Snapshots: every successful `PUT` writes `storage/architectures/{id}/v{version}.json` before bumping the
row. Generated code and compile output live in the run dir, never in git.

## Data Flow

```
Canvas edit
  → graph IR (client state)
  → POST /validate            → analytic shapes + issues, rendered on nodes and in the Issues tab
  → PUT  /architectures/{id}  → DB row + storage snapshot
  → POST /compile             → subprocess: emit .py, import, build, model.summary()
  → GET  /code                → generated_model.py, shown in the Code tab and downloadable
  → POST /api/training/jobs   (model_id="architecture_graph", hyperparameters.architecture_id=...)
        → service emits generated_model.py into run_dir
        → architecture_train.py imports it, builds, trains
        → best_model.keras / last_model.keras / metrics.json / results.csv in run_dir
        → promote → model_registry as source="trained"
        → /testing, /inference, /models export all work unchanged
```

The graph is the source of truth at every step. Generated code is always a pure function of the graph
and is regenerated rather than stored, so a hand-edit to exported code never silently diverges from what
trains. Editing exported code is a one-way door: the user owns the file from that point.

### Validation tiers

1. **Analytic** (`/validate`, synchronous, no TF). Structural checks — DAG-ness, exactly one reachable
   input and output path, port arity, orphan nodes — plus per-node shape propagation from rules declared
   on each `NodeSpec`. Fast enough to run on every canvas change, debounced.
2. **Compile** (`/compile`, subprocess). Emits the module, imports it, builds the model, returns the real
   summary. This is authoritative and catches anything the analytic pass approximates. It is a
   subprocess for the same reason the training runners are: TensorFlow must not be imported by the API
   process, per `AGENTS.md`.

### Code generation

Deterministic. Variable names derive from node ids, so regenerating an unchanged graph produces a
byte-identical file and diffs are meaningful.

```python
# Generated by Onestep AI Platform — architecture "small-cnn" (arch_7f2a, v4)
# Edits to this file are not read back into the studio.
import tensorflow as tf


def build_model(num_classes: int = 2, input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x_conv_1 = tf.keras.layers.Conv2D(32, 3, padding="same", name="conv_1")(inputs)
    x_bn_1 = tf.keras.layers.BatchNormalization(name="bn_1")(x_conv_1)
    x_act_1 = tf.keras.layers.Activation("relu", name="act_1")(x_bn_1)
    x_pool_1 = tf.keras.layers.GlobalAveragePooling2D(name="pool_1")(x_act_1)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="head")(x_pool_1)
    return tf.keras.Model(inputs, outputs, name="small_cnn")


if __name__ == "__main__":
    build_model().summary()
```

Custom layer classes are hoisted above `build_model`. `num_classes` is a parameter rather than a literal
so the same architecture retrains against a different dataset without editing the graph; the runner
passes the dataset's label count.

## Node Catalog

| Category | Nodes |
| --- | --- |
| I/O | Input, Output |
| Core | Dense, Activation, Dropout, Flatten, Reshape, Permute |
| Convolution | Conv1D, Conv2D, SeparableConv2D, Conv2DTranspose, MaxPool2D, AvgPool2D, GlobalAvgPool2D, GlobalMaxPool2D, UpSampling2D, ZeroPadding2D |
| Normalization | BatchNormalization, LayerNormalization, GroupNormalization, RMSNorm |
| Recurrent | LSTM, GRU, Bidirectional |
| Merge | Add, Concatenate, Multiply, Subtract, Average |
| Transformer | Embedding, PositionalEmbedding, RotaryPositionalEmbedding, MultiHeadAttention (with `causal` flag), FeedForward, SwiGLU, CausalMask, LMHead |
| Regularization | SpatialDropout2D, GaussianNoise, ActivityRegularization |
| Backbone | PretrainedBackbone — wraps `tf.keras.applications.*` with `include_top=False`, reusing `KERAS_APPLICATION_OPTIONS`, so a graph can start from transfer learning |
| Structure | Group (subgraph with a `repeat` count) |
| Custom | CustomLayer, CustomFunction |

`RMSNorm`, `RotaryPositionalEmbedding`, and `SwiGLU` have no stock Keras layer and emit small generated
classes from templates in `emit_keras.py`.

The Group node is what makes from-scratch transformers tractable: author one block
(RMSNorm → MHA(causal) → residual → RMSNorm → SwiGLU → residual), set `repeat: 12`, and codegen emits a
loop rather than twelve copies of the same subgraph.

## Custom code and trust

A CustomLayer node holds a Python class body; a CustomFunction node holds an expression body wrapped in
a `Lambda`. Both are written verbatim into the generated module.

- Custom code is **never** imported or executed by the FastAPI process. It runs only in the compile and
  training subprocesses, which is exactly how every existing runner already works.
- This is arbitrary code execution by design, appropriate for the current single-user local deployment
  and no further. `docs/ai/rules.md` already requires auth before multi-user or network-exposed
  deployment; this feature makes that requirement load-bearing rather than precautionary, and the
  studio surfaces a one-line notice on the custom node saying the code runs locally with full
  permissions.
- Compile failures surface the real traceback, trimmed to the generated file's frames, so a syntax error
  in a custom node points at the node rather than at the runner.
- Custom code is part of the graph JSON, so it travels through export, import, and snapshots. Importing
  a `.json` from an untrusted source imports its code; the import dialog says so and shows the custom
  node bodies before the import is confirmed.

## Import

| Source | Behavior |
| --- | --- |
| Platform `.json` | Validated against `schema_version`, positions preserved. |
| Keras `.keras` / `model.to_json()` | Config parsed in a subprocess, layers mapped to catalog node types, unmapped layers become CustomLayer nodes carrying their config so nothing is silently dropped. Auto-layout assigns positions. |
| Registry `model_id` | Same path, reading the registered `.keras` artifact. Lets a user open a trained or uploaded model, see its architecture, fork it, and retrain. |

Auto-layout is a pure layered pass — longest-path layering, in-layer ordering by median predecessor
position, fixed column and row pitch. No new dependency.

## Edge Cases

- **Cyclic graph** — analytic pass reports the participating node ids; save is allowed, compile and
  train are blocked.
- **Disconnected / orphan nodes** — warning, excluded from codegen, node dimmed on canvas.
- **No Input or no Output node** — error; train blocked.
- **Shape mismatch on a merge node** — analytic pass flags it on the merge node with both incoming
  shapes in the message.
- **Unknown node type in an imported graph** — imported as a disabled placeholder node with its raw
  params retained, flagged as an error, rather than dropped.
- **Stale save** — `PUT` with an older `version` returns 409 and the studio offers reload or overwrite.
- **Compile timeout** — subprocess capped at 120s; timeout is reported as a compile failure.
- **Custom code raising at build time** — traceback surfaced in the Issues tab and the toast layer, per
  `docs/ai/rules.md` "avoid hiding errors".
- **Training a graph whose architecture changed mid-run** — the run dir holds the emitted module, so a
  running job is pinned to the graph as of job creation. The job record stores the architecture id and
  version.
- **Deleting an architecture that trained models** — allowed; registered models keep their weights and
  their `architecture_id` metadata, which resolves to "deleted" in the UI.
- **`num_classes` mismatch** — the runner passes the dataset's label count; a graph whose Output node
  hardcodes a different unit count is rejected before training starts with an explicit message.
- **Very large graphs** — canvas virtualizes above 200 nodes; validation is debounced at 300ms.

## Implementation Stages

Ordered so each stage is independently shippable and testable.

- **A. Foundation (backend, no UI). Done.** IR, node catalog, DAG validation, analytic shape inference,
  Keras emitter, auto-layout, starter templates, CRUD service, migration, router. 109 tests, of which
  5 are `slow`-marked and build the generated modules under real TensorFlow — those assert that every
  predicted shape equals the built layer's actual shape and that the parameter estimate matches
  `model.count_params()` exactly.
- **B. Studio UI.** React Flow canvas, palette, inspector via the phase 13 field renderer, code panel,
  summary panel, save/load, export, import, auto-layout, templates.
- **C. Training.** Extract `keras_common.py`, add `architecture_graph` to the training catalog, add
  `architecture_train.py`, wire promotion into the registry, add the studio's Train action prefilling
  the existing training form.
- **D. Custom code. Done.** `custom_layer` (a hoisted Keras subclass) and `custom_function` (a
  single expression in a Lambda), a `code` param type rendered as a mono textarea, per-class
  hoisting, and optional declared output shapes.
- **E. Transformer primitives and causal LM. Done.** RoPE, RMSNorm, SwiGLU, feed-forward, causal
  multi-head attention, learned positional embeddings, LM head, GlobalAvgPool1D, and a composite
  `transformer_block` with a repeat count. Plus the `architecture_lm` family: a word/char tokenizer
  over a text corpus, next-token windowing, perplexity, and temperature-sampled generations written
  to `sample_generations.json`.

## Acceptance Criteria

Behavior:

- A user creates an architecture from the Models page, drags Input → Conv2D → BatchNorm → ReLU →
  GlobalAvgPool → Dense onto the canvas, wires it, sees per-node output shapes and a total parameter
  count, saves it, and trains it against a project dataset from the existing training surface.
- The resulting model appears in the model catalog as a trained model and is usable on `/testing` and
  `/inference` with no code path specific to graph-built models.
- Exported `generated_model.py` runs standalone: `python generated_model.py` prints a model summary.
- Exporting an architecture to `.json` and importing that file reproduces an identical graph, positions
  included.
- Importing a trained `.keras` model produces an editable graph whose compile summary matches the
  original model's layer count and parameter count.
- A CustomLayer node with a valid Keras layer subclass compiles, trains, and round-trips through export
  and import. An invalid one reports its traceback against that node.
- A decoder-only transformer built from Embedding + RoPE + causal MHA + RMSNorm + SwiGLU + LMHead, with
  a Group node repeated N times, compiles and trains to decreasing loss on a small text dataset.

Tests:

- `backend/tests/test_architecture_graph.py` — parse, topological sort, cycle detection, port arity.
- `backend/tests/test_architecture_shapes.py` — analytic shapes across conv/pool/merge/attention chains,
  including deliberate mismatches.
- `backend/tests/test_architecture_emit.py` — golden-file generated source for fixture graphs, including
  a Group repeat and a custom node.
- `backend/tests/test_architecture_api.py` — CRUD, stale-version 409, validate, import from `.json`.
- `backend/tests/test_architecture_train.py` — the runner builds and trains a tiny graph for one epoch on
  a synthetic dataset, and writes the artifacts `_register_training_model` expects.
- `frontend/features/models/architectures/*.test.tsx` — graph reducer, auto-layout determinism, palette
  filtering by task type.

Manual checks:

- Studio renders correctly in light and dark, uses no hardcoded color, and adds no `box-shadow` to a
  resting panel.
- Canvas is keyboard-navigable: node selection, delete, and inspector focus without a pointer.
- Every backend failure surfaces in both the Issues panel and the toast layer.

## Validation

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm test
cd frontend && pnpm build
```

Plus one end-to-end smoke run: build a small CNN in the studio, train one epoch against a local dataset,
and confirm the promoted model runs on `/inference`.

## Open Questions

- Whether trained-model → architecture import should also reconstruct weights into the studio, or stay
  architecture-only. Drafted as architecture-only; weights stay with the registered model.
- Whether `language_modeling` should be added to the project task-type vocabulary or stay internal to
  the architecture studio until a `keras_lm` predictor exists.
