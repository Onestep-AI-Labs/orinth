# Spec: Phase 1 Inference

## Status

Implemented MVP

## Goal

Serve dental radiograph inference for both available model families:

- YOLOv11 segmentation
- U-Net lesion segmentation plus Inception classification

## Interfaces

- `GET /health`
- `GET /api/models`
- `POST /api/inference`
- `GET /api/inference`
- `GET /api/inference/{id}`

## Behavior

- Models are discovered from local `models/`.
- Predictors are lazy-loaded.
- Uploads and overlays are stored under ignored `storage/`.
- Both model families return the same `InferenceResult` shape.
- No detections returns image-level `Normal`.

## Model Defaults

- YOLO weights: `models/yolo_11_best/weights/best.pt`
- U-Net: `models/unet_inception/best_unet_model.keras`
- Inception classifier: `models/unet_inception/best_classifier_inception.keras`
- Confidence threshold: `0.65`
- Inference IoU threshold: `0.7`

## Acceptance Criteria

- Both models appear in `GET /api/models`.
- Uploading an image returns detections, normalized boxes/polygons, image label, and overlay URL.
- Frontend can choose either model and display result/history.
- Backend tests pass.
- At least one local smoke inference passes for each available model.
