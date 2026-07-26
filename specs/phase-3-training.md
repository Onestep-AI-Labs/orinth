# Spec: Phase 3 Training

## Status

YOLO observable job MVP; Keras image classification MVP; NLP Keras and Hugging Face transformer paths in progress

## Goal

Provide a local, project-scoped training job platform that runs long image training work outside the FastAPI request path.

## Interfaces

- `POST /api/training/jobs`
- `GET /api/training/jobs`
- `GET /api/training/jobs/{id}`
- `POST /api/training/jobs/{id}/cancel`
- `POST /api/training/jobs/{id}/promote`
- `GET /api/training/model-options?task_type=...`
- `POST /api/training/model-assets/prepare`
- `POST /api/training/jobs/delete`
- `DELETE /api/training/jobs`
- `DELETE /api/training/jobs/{id}`
- `PATCH /api/models/{model_id}`
- `DELETE /api/models/{model_id}`

## Behavior

- Training jobs run in subprocesses.
- Logs and artifacts are written under ignored `storage/training_runs`.
- Jobs include `project_id`, `task_type`, `model_option_id`, optimizer, architecture, and hyperparameter payloads.
- Training job creation accepts an optional model display name, which is applied to the auto-registered completed model artifact.
- Training creation is task-first: users choose classification, object detection, or segmentation before selecting a compatible model option.
- The frontend defaults new training jobs to classification and initializes the first runnable classification model option when available.
- YOLO training uses local dataset paths, dataset-driven labels, and notebook-derived hyperparameters.
- YOLO training accepts dataset id, device, cache mode, workers, patience, optimizer, and learning rate.
- YOLO logs stream into job artifacts while the subprocess runs.
- YOLO jobs expose progress, elapsed time, log tail, and parsed `results.csv` metrics when available.
- Training progress uses parsed epoch history from `results.csv` where available, exposing `processed/total` as current epoch over requested epochs.
- Progress is one unified 0–100 bar across the **whole** run — for LLM runs, data prep → base-model download → weight load → training loop → saving each own a contiguous band (`artifacts.llm_progress`); epoch-denominated runs use a small prep band then the epoch fraction (`artifacts.epoch_progress`). 100% means the run has finished, not just that the training loop reached its last step. The reported percent is clamped monotonic while `status == "running"` so a truncated log window or a dip in the self-reported download percentage never walks the bar backwards. This replaces the earlier `min(99, 5 + len(logs))` fallback that raced the download log count to 99% while the multi-GB base was still at 43%.
- The base-model download band is driven by the runner's own `downloading base model: N%` line, not by log-line count, so the bar tracks real bytes transferred.
- Runners therefore rewrite `results.csv` after **every** epoch, not once at the end. A runner that writes it only on completion leaves the progress bar pinned and the curves empty for the whole run.
- Training subprocesses run with `PYTHONUNBUFFERED=1`. `Popen(bufsize=1)` only line-buffers the parent's read of the pipe; without the env var the child block-buffers its own stdout and per-epoch output arrives in a single burst at exit, so the live log appeared to jump from checkpoint loading straight to completion.
- Keras Applications classification training runs through a TensorFlow subprocess runner and writes `results.csv`, `best_model.keras`, `last_model.keras`, and `metrics.json`.
- Completed runnable YOLO and Keras classification jobs are copied to stable ignored trained-model directories named with both a slugified model display name and the stable model id, then registered automatically.
- Training jobs expose full metric history, curve metadata, public artifact URLs, and latest metrics.
- Keras classification runs write validation predictions, confusion matrix, classification report, ROC curve data, and macro/micro AUC when computable.
- Dataset preprocessing is applied at training time by preparing transformed copies under the training run directory; original dataset images are never overwritten.
- Preprocessing uses allowlisted Albumentations transforms with image-only classification, bbox-aware detection, and mask/polygon-compatible segmentation handling where applicable.
- Model options list local YOLO, Ultralytics YOLO11/YOLO26 runnable entries, Keras Applications CNNs, task-declared NLP baselines, Keras NLP models, runnable Hugging Face NLP transformers, and gated unvalidated specialist families.
- Runnable Ultralytics YOLO options use the YOLO family runner with weights such as `yolo11n.pt`, `yolo11n-seg.pt`, `yolo26n.pt`, and `yolo26n-seg.pt`.
- Ultralytics SAM3, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, and YOLO-World are cataloged but gated until their dataset and runner flows are validated.
- Keras Applications options follow the official constructor pattern: `weights="imagenet"`, `include_top=False`, `pooling="avg"`, dataset-driven `input_shape`, and option-specific constructor kwargs such as MobileNetV2 `alpha` or EfficientNetB7 `name`.
- Keras, Ultralytics, and Hugging Face asset preparation is explicit; unvalidated specialist preparation returns a gated status.
- Hugging Face asset preparation uses `HUGGINGFACE_HUB_TOKEN`/`HF_TOKEN` when configured and prepares BERT backbone assets without instantiating task heads.
- NLP model options are discovered from task/model package catalogs instead of being hardcoded directly in the training service.
- Runnable NLP Keras text-classification jobs write `best_model.keras`, `last_model.keras`, `tokenizer.json`, `metadata.json`, `results.csv`, `metrics.json`, and `validation_predictions.json`.
- Runnable Hugging Face NLP jobs write a saved model directory plus tokenizer/config files and run-level metrics/prediction artifacts.
- Hugging Face BERT training suppresses expected task-head load reports and logs a concise note that task heads are initialized for the selected dataset.
- Hyperparameter fields hydrate from the selected option's catalog defaults. The learning-rate field surfaces the recommended value only once it differs from what is entered — at the default the field already holds that value, so restating it is noise. For Hugging Face options a rate above `1e-4` is flagged as too high: transformer fine-tuning needs roughly `1e-5`–`5e-5`, and an order-of-magnitude larger rate wrecks the pretrained weights, producing a validation curve that oscillates and rises while training loss falls. The value is still accepted — it is a warning, not a block.
- Training UI and detail views show task-relevant parameters; NLP length and vocabulary settings are sent through `hyperparameters`.
- Terminal jobs can be deleted singly, in selected batches, or by clear-all; active jobs are blocked.
- Backend startup marks queued/running jobs as failed because FastAPI background subprocesses do not survive restarts.
- Completed YOLO and Keras classification runs can be used directly from the model registry; manual promotion remains idempotent.
- Trained and promoted model registry entries can be renamed or deleted from the model catalog; reference models remain read-only.
- U-Net + Inception training is gated until the long two-stage TensorFlow path is validated.

## Acceptance Criteria

- Training dashboard can create and list jobs.
- Training dashboard filters model options by selected task before allowing job creation.
- Training dashboard defaults to classification and keeps model selection synchronized with the selected task.
- YOLO runner writes logs, progress, parsed metrics, and artifacts.
- Keras classification runner command generation is tested for official application examples and runnable when an annotated classification dataset is available.
- Hugging Face token aliases are exported to subprocess training environments when configured.
- Training forms and detail pages do not show image-only parameters for NLP jobs.
- Active YOLO jobs can be canceled.
- Failed jobs persist a clear error.
- Promotion registers a completed YOLO `best.pt` as an inference model.
- Successful runnable training jobs appear in Available Models without requiring a manual promote click.
- The training detail page leads with the unified progress bar (percent-primary, ETA when known) then a **live metrics** grid — the values that change during a run (progress, elapsed, ETA, step/epoch, loss, val loss, best val loss, learning rate, and the family's headline metric: mAP / accuracy / F1 / ROUGE / perplexity). Static Parameters and Advanced settings move below the graphs, not above them.
- Metric history renders as interactive `TrainingChart` panels (`features/training/training-chart.tsx`): a self-contained SVG line chart with axes, gridlines, a legend, and a hover crosshair + per-series tooltip. Series are grouped by a shared Y scale (loss curves together, mAP curves together, detection losses, learning rate, …) so overlaid lines stay comparable; only groups with an available column render.
- The run log is shown once, on the detail page, with consecutive duplicate lines collapsed (`dedupeConsecutive`); `ProgressPanel` takes `hideLogs` there so the same lines are not repeated inline under the bar.
- `ProgressPanel` shows the unified percent as the primary readout with the step/epoch count as a secondary detail, so the header can never contradict the bar.
- Training list/detail progress displays epoch counts when the runner has produced epoch metrics.
- Frontend lists runs separately from a dedicated training detail page.

## Deferred

- Full U-Net + Inception two-stage training runner.
- Additional Hugging Face model families beyond initial BERT text classification/QA and BART summarization.
