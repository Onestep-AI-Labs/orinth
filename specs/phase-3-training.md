# Spec: Phase 3 Training

## Status

YOLO observable job MVP; Keras classification runner MVP; U-Net + Inception and transformer paths gated

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

## Behavior

- Training jobs run in subprocesses.
- Logs and artifacts are written under ignored `storage/training_runs`.
- Jobs include `project_id`, `task_type`, `model_option_id`, optimizer, architecture, and hyperparameter payloads.
- Training creation is task-first: users choose classification, object detection, or segmentation before selecting a compatible model option.
- The frontend defaults new training jobs to classification and initializes the first runnable classification model option when available.
- YOLO training uses local dataset paths, dataset-driven labels, and notebook-derived hyperparameters.
- YOLO training accepts dataset id, device, cache mode, workers, patience, optimizer, and learning rate.
- YOLO logs stream into job artifacts while the subprocess runs.
- YOLO jobs expose progress, elapsed time, log tail, and parsed `results.csv` metrics when available.
- Training progress uses parsed epoch history from `results.csv` where available, exposing `processed/total` as current epoch over requested epochs.
- Keras Applications classification training runs through a TensorFlow subprocess runner and writes `results.csv`, `best_model.keras`, `last_model.keras`, and `metrics.json`.
- Completed runnable YOLO and Keras classification jobs are copied to stable ignored `storage/trained_models/{model_id}` directories and registered automatically.
- Training jobs expose full metric history, curve metadata, public artifact URLs, and latest metrics.
- Keras classification runs write validation predictions, confusion matrix, classification report, ROC curve data, and macro/micro AUC when computable.
- Dataset preprocessing is applied at training time by preparing transformed copies under the training run directory; original dataset images are never overwritten.
- Preprocessing uses allowlisted Albumentations transforms with image-only classification, bbox-aware detection, and mask/polygon-compatible segmentation handling where applicable.
- Model options list local YOLO, Ultralytics YOLO11/YOLO26 runnable entries, Keras Applications CNNs, gated Ultralytics specialist families, and gated Hugging Face transformer entries.
- Runnable Ultralytics YOLO options use the YOLO family runner with weights such as `yolo11n.pt`, `yolo11n-seg.pt`, `yolo26n.pt`, and `yolo26n-seg.pt`.
- Ultralytics SAM3, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, and YOLO-World are cataloged but gated until their dataset and runner flows are validated.
- Keras Applications options follow the official constructor pattern: `weights="imagenet"`, `include_top=False`, `pooling="avg"`, dataset-driven `input_shape`, and option-specific constructor kwargs such as MobileNetV2 `alpha` or EfficientNetB7 `name`.
- Keras, Ultralytics, and Hugging Face asset preparation is explicit; unvalidated specialist and transformer preparation returns a gated status.
- Terminal jobs can be deleted singly, in selected batches, or by clear-all; active jobs are blocked.
- Backend startup marks queued/running jobs as failed because FastAPI background subprocesses do not survive restarts.
- Completed YOLO and Keras classification runs can be used directly from the model registry; manual promotion remains idempotent.
- U-Net + Inception training is gated until the long two-stage TensorFlow path is validated.

## Acceptance Criteria

- Training dashboard can create and list jobs.
- Training dashboard filters model options by selected task before allowing job creation.
- Training dashboard defaults to classification and keeps model selection synchronized with the selected task.
- YOLO runner writes logs, progress, parsed metrics, and artifacts.
- Keras classification runner command generation is tested for official application examples and runnable when an annotated classification dataset is available.
- Active YOLO jobs can be canceled.
- Failed jobs persist a clear error.
- Promotion registers a completed YOLO `best.pt` as an inference model.
- Successful runnable training jobs appear in Available Models without requiring a manual promote click.
- Training detail pages show metric history charts and runner curve artifacts.
- Training list/detail progress displays epoch counts when the runner has produced epoch metrics.
- Frontend lists runs separately from a dedicated training detail page.

## Deferred

- Full U-Net + Inception two-stage training runner.
- Runnable Hugging Face transformer training.
