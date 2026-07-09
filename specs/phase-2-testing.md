# Spec: Phase 2 Testing

## Status

Implemented MVP

## Goal

Run local evaluation jobs against test splits and store metrics/artifacts.

## Interfaces

- `GET /api/testing/datasets`
- `POST /api/testing/jobs`
- `GET /api/testing/jobs`
- `GET /api/testing/jobs/{id}`

## Datasets

- YOLO test split: `datasets/dental dataset_yolov11_format/test`
- COCO test split: `datasets/dental dataset_coco_format/test`

## Metrics

- Pixel: Dice, IoU, precision, recall, specificity.
- Object: bbox IoU matching at threshold `0.2`, confusion matrix with background row/column.
- Image: majority ground-truth object count versus predicted largest mask area.
- Classification: accuracy, balanced accuracy, macro F1, weighted F1, sensitivity, specificity, LR+/LR-, Cohen kappa, MCC.

## Acceptance Criteria

- Testing dashboard can create jobs.
- Jobs persist status, metrics, artifacts, and errors.
- Evaluation can run either registered predictor against local test images.
- Metrics are stored under ignored `storage/evaluations`.
