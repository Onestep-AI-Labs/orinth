# Spec: Phase 1 Inference

## Status

Implemented with observable job MVP and project-scoped history

## Goal

Serve local image inference for registered model families. The original reference models remain available:

- YOLOv11 segmentation
- U-Net lesion segmentation plus Inception classification

## Interfaces

- `GET /health`
- `GET /api/models?available_only=false`
- `GET /api/models/{model_id}/download`
- `POST /api/inference`
- `POST /api/inference/jobs`
- `GET /api/inference/jobs/{id}`
- `GET /api/inference`
- `GET /api/inference/{id}`
- `POST /api/inference/delete`
- `DELETE /api/inference`
- `DELETE /api/inference/{id}`

## Behavior

- Models are discovered from local `models/`.
- Models also include stable trained artifacts registered under ignored `storage/trained_models`.
- Model listing supports project and task filters and returns task type, labels, source, metrics, and training job metadata.
- Model catalog entries can be downloaded from `GET /api/models/{model_id}/download`; single-file models stream the asset directly, and multi-asset models are bundled as zip archives under ignored storage.
- Inference UI queries `GET /api/models?available_only=true`; users can only run models with local or promoted trained assets available.
- Inference UI selection is task-first, then available models filtered to the selected task.
- Inference UI renders task-aware parameters: classification models hide detection-specific thresholds, while detection and segmentation models expose confidence and IoU controls.
- Inference UI uses neutral inference iconography because the page supports both image and text inference.
- Inference UI keeps model task metadata and result summary as compact inline rows instead of boxed metric grids.
- Predictors are lazy-loaded.
- YOLO predictors use registry-provided labels so trained project models are not limited to dental labels.
- Keras classification models return image-level labels and class scores with no object detections.
- Text inference supports baseline NLP models, trained Keras NLP artifacts, and trained Hugging Face NLP artifact directories through normalized `nlp_result` payloads.
- Uploads and overlays are stored under ignored `storage/`.
- Both model families return the same `InferenceResult` shape.
- No detections returns image-level `Normal`.
- The original synchronous inference endpoint remains compatible.
- Inference requests accept optional `project_id`; old requests default to the default research project.
- Inference history can be filtered by project.
- Selected or clear-all inference history deletion removes DB rows plus owned uploads/overlays under `storage/`.
- The job endpoint persists status, progress, current step, logs, elapsed time, errors, and final result.
- Inference results may include `duration_ms` and step `timings`.

## Model Defaults

- YOLO weights: `models/yolo_11_best/weights/best.pt`
- U-Net: `models/unet_inception/best_unet_model.keras`
- Inception classifier: `models/unet_inception/best_classifier_inception.keras`
- Confidence threshold: `0.65`
- Inference IoU threshold: `0.7`

## Acceptance Criteria

- Both models appear in `GET /api/models`.
- Models can be downloaded from the model catalog options menu when their local assets are available.
- Inference forms list only available trained/registered models.
- Completed runnable training jobs appear as available models for inference.
- Completed NLP Keras/Hugging Face training jobs appear as available text models for compatible tasks.
- Uploading an image returns detections, normalized boxes/polygons, image label, and overlay URL.
- Job-based inference shows upload/model/prediction/overlay/persistence progress.
- Frontend can choose a project/model and display result/history in the sidebar app.
- Frontend inference parameters match the selected model task.
- Inference result empty states use the same neutral icon as the inference sidebar entry.
- Frontend can delete selected inference history rows or clear project history.
- Backend tests pass.
- At least one local smoke inference passes for each available model.
