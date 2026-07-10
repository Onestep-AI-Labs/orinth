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
- Inference UI queries `GET /api/models?available_only=true`; users can only run models with local or promoted trained assets available.
- Inference UI renders task-aware parameters: classification models hide detection-specific thresholds, while detection and segmentation models expose confidence and IoU controls.
- Predictors are lazy-loaded.
- YOLO predictors use registry-provided labels so trained project models are not limited to dental labels.
- Keras classification models return image-level labels and class scores with no object detections.
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
- Inference forms list only available trained/registered models.
- Completed runnable training jobs appear as available models for inference.
- Uploading an image returns detections, normalized boxes/polygons, image label, and overlay URL.
- Job-based inference shows upload/model/prediction/overlay/persistence progress.
- Frontend can choose a project/model and display result/history in the sidebar app.
- Frontend inference parameters match the selected model task.
- Frontend can delete selected inference history rows or clear project history.
- Backend tests pass.
- At least one local smoke inference passes for each available model.
