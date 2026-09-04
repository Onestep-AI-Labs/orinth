# Spec: Phase 24 Compute Targets

## Status

**Implemented for local compute.** Remote providers are declared and not built — see Deferred.

Landed:

- `backend/app/ml/compute.py` — a per-framework device probe, run in a subprocess and cached.
- `backend/app/ml/compute_plan.py` — resolves `device: auto` into a device, batch size, and
  precision, each with its reason.
- `backend/app/ml/compute_providers.py` — the provider registry; `local` available, three declared.
- `GET /api/training/compute`, `/compute/plan`, `/compute/providers`.
- `TrainingService._device_env` resolves once and hands the answer to every runner through
  `ONESTEP_TRAIN_DEVICE`.
- `orinth doctor` prints the per-framework table and the notes.

## Goal

Make a training run adapt to the hardware it is actually on, and say what it decided before it
starts — rather than discovering it from a run that is twenty times slower than expected, or from
an allocator error forty minutes in that names bytes instead of the setting that caused them.

## The decisions

### The unit of truth is a (framework, device) pair

Phase 14's `probe_llm_environment` reports one `device` field. That is correct for `llm_sft` and
wrong for the machine, and this machine proves it: **torch reports MPS available while TensorFlow
reports CPU only**, because `tensorflow-metal` is a separate wheel that is not installed. A user
here gets a GPU for LLM fine-tuning and silently gets CPU for Keras training.

A single `device` field cannot express that. So each backend is probed on its own terms and
`FRAMEWORK_FOR_FAMILY` maps a model family to the framework that backs it — both halves are needed
to answer "will this particular run use the GPU".

The disagreement is surfaced as a note naming the fix, because it is invisible otherwise and nobody
guesses the cause.

### The probe runs out of process, and is cached

Importing torch and TensorFlow into the API to ask them about devices is what `docs/ai/rules.md`
forbids, and the cost is real: TF alone is seconds of import and hundreds of megabytes resident,
paid by a worker that will never train anything. The probe is `sys.executable -c` returning one
JSON line, cached for ten minutes — a machine does not grow a GPU between requests, and a restart
is exactly when it could.

### Batch size adapts; an explicit one is never lowered

`plan_for` derives a batch size from reported memory minus 25% headroom, divided by a per-family
per-sample footprint, rounded down to a power of two and capped at 64. Every number is labelled an
estimate wherever it surfaces, because it is one — the real figure depends on image size, sequence
length, and the checkpoint.

Two refusals matter more than the arithmetic:

- **An explicit batch size is warned about, never lowered.** An override is a decision, and
  silently halving it would make the form lie about what ran.
- **Unknown memory keeps the family default.** TensorFlow reports no VRAM; Apple silicon reports a
  ceiling on a pool shared with everything else on the machine. Deriving a batch size from a figure
  nobody has is how a default becomes confidently wrong.

### Precision follows the hardware

bf16 needs compute capability 8.0+ (Ampere); on older CUDA cards fp16 is the one that actually
speeds anything up. Metal gets fp16 because bf16 has been unreliable across torch releases. CPU
gets fp32 because reduced precision on a CPU is slower, not faster.

### One resolution point, not one per runner

The resolved device reaches every runner through `ONESTEP_TRAIN_DEVICE` in the subprocess
environment. Two runners already read it, so this needed no change to any command builder — and it
means the device a run used is decided in one place rather than re-derived, differently, by each
runner. A probe failure is swallowed: every runner keeps its own `cuda > mps > cpu` default, which
is exactly the behaviour that existed before.

## Interfaces

- `GET /api/training/compute?refresh=` → `ComputeEnvironment`
- `GET /api/training/compute/plan?task_type=&model_family=&device=&batch_size=` → `ComputePlan`
- `GET /api/training/compute/providers` → `list[ComputeProvider]`
- Schemas: `ComputeKind`, `ComputeDevice`, `ComputeFramework`, `ComputeEnvironment`, `ComputePlan`,
  `ComputeProviderId`, `ComputeProvider`.
- No storage or DB change.

## Edge cases

- **Probe times out** (90s) → `ComputeEnvironment.error`, and the planner falls back to the family
  default rather than blocking a run.
- **A framework is not importable** → its `installed: false` with the error; other frameworks are
  unaffected, which is why the error lives per framework rather than on the environment.
- **A requested device the probe cannot see** → honoured with a warning. The user may know
  something the probe does not.
- **No GPU at all** → a plain note. A CPU run is a real answer, not a failure.

## Acceptance criteria

- A machine where torch and TensorFlow disagree produces a note naming the fix. ✅ (verified on the
  development machine: `Install tensorflow-metal to give Keras the same GPU`)
- `orinth doctor` shows the per-framework table. ✅
- A Keras family on a torch-only-GPU machine plans a CPU run and warns. ✅
- Batch size scales with memory and is capped. ✅
- 21 tests in `backend/tests/test_compute.py`.

## Deferred

**Remote GPU providers — Vast.ai, Modal, RunPod.** Declared in `compute_providers.py` with
`available: false` and concrete `requirements`, listed rather than hidden so the roadmap is visible
and so `available: false` is a fact the client reads rather than a string it hardcodes.

The requirements are the implementation checklist, and they are mostly shared:

1. **Credentials in settings**, stored like the OpenRouter key — never returned to a client.
2. **Dataset transfer.** Remote instances have their own disk; `training_root` resolves a local
   path today, so a run needs the split uploaded rather than a shared filesystem.
3. **Artifact retrieval**, since an instance is destroyed when the run ends and
   `promote_model` reads a local directory.
4. **Log streaming** into the job's log tail, so a remote run reads the same as a local one.
5. **A cost ceiling per job.** Billing by the second with no cap is how a training run becomes an
   invoice nobody approved. This is the requirement most likely to be skipped and least acceptable
   to skip.

Also deferred: measuring the per-sample footprints rather than estimating them, multi-GPU
(`device_count > 1` is detected and reported but only device 0 is used), and ROCm (the probe has a
`rocm` kind but nothing sets it — no hardware to verify against, and a code path nobody has run is
not support).

## Validation

```bash
cd backend && uv run ruff check .
cd backend && uv run pytest
cd frontend && pnpm typecheck && pnpm lint && pnpm test && pnpm build
```
