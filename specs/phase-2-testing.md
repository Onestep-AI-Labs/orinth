# Spec: Phase 2 Testing

## Status

Implemented with observable job MVP and project-scoped deletion

## Goal

Run local evaluation jobs against image dataset splits and store metrics/artifacts.

## Interfaces

- `GET /api/testing/datasets`
- `POST /api/testing/jobs`
- `POST /api/testing/jobs/batch`
- `GET /api/testing/jobs`
- `GET /api/testing/jobs/{id}`
- `GET /api/testing/jobs/{id}/comparison`
- `GET /api/testing/jobs/{id}/per-image`
- `POST /api/testing/jobs/delete`
- `DELETE /api/testing/jobs`
- `DELETE /api/testing/jobs/{id}`

## Datasets

- YOLO test split: `datasets/dental dataset_yolov11_format/test`
- COCO test split: `datasets/dental dataset_coco_format/test`
- Editable project datasets with YOLO-compatible image splits.
- Editable classification datasets with processed train/valid/test splits.
- Editable project datasets are offered to testing only through their `test` split; train/valid splits are excluded from testing dataset selection.

## Metrics

- Pixel: Dice, IoU, precision, recall, specificity.
- Object: bbox IoU matching at threshold `0.2`, confusion matrix with background row/column.
- Image: majority ground-truth object count versus predicted largest mask area.
- Classification: accuracy, balanced accuracy, macro F1, weighted F1, sensitivity, specificity, LR+/LR-, Cohen kappa, MCC.
- Classification models with score outputs also store ROC curves and macro/micro AUC when valid probabilities are available.
- Frontend testing tables choose visible metric columns from the evaluated task: classification emphasizes accuracy/F1/MCC/AUC, while detection and segmentation emphasize pixel and object metrics.

## Acceptance Criteria

- Testing dashboard can create jobs.
- Testing dashboard selection is task-first, then compatible models, then compatible datasets.
- Testing dashboard can select multiple available models and create one evaluation job per model for comparison.
- Testing only accepts model/dataset task-compatible pairs.
- Keras classification trained models can be evaluated on classification dataset splits.
- Jobs persist status, metrics, artifacts, and errors.
- Batch-created testing jobs share a `comparison_id`; single jobs use their own job id as the stable `comparison_id`.
- Testing detail pages can load any job and compare all sibling jobs in the same comparison group.
- Testing detail pages show metric summary cards, task-relevant confusion matrices, object summaries, and task-aware per-image tables without extra chart panels.
- Testing comparison views use task-aware metric tables for completed sibling jobs.
- Jobs include `project_id`; list and dataset APIs support project filtering.
- Metrics use labels from the selected dataset rather than fixed dental constants.
- Terminal jobs can be deleted singly, in selected batches, or by clear-all; active jobs are blocked.
- Jobs expose progress with processed/total samples, percent, current image, elapsed time, ETA, and logs.
- Per-image details follow the evaluated task: classification rows show ground-truth class, predicted class, and accuracy score; detection and segmentation rows include labels, pixel metrics, and object matching summary.
- Frontend polling is adaptive and stops when jobs are terminal.
- Frontend lists jobs separately from a dedicated testing detail page.
- Frontend shows a compact comparison table for all jobs in the selected comparison group.
- Frontend testing dataset selection lists only reference test splits and editable dataset test splits.
- Evaluation can run either registered predictor against local test images.
- Metrics are stored under ignored `storage/evaluations`.
