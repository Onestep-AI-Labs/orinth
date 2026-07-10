# Spec: Phase 4 Dataset Studio

## Status

Implemented generalized image MVP

## Goal

Provide a local dataset workspace for browsing reference datasets and managing editable image datasets.

## Interfaces

- `GET /api/datasets`
- `POST /api/datasets`
- `PATCH /api/datasets/{id}`
- `POST /api/datasets/import`
- `POST /api/datasets/{id}/clone`
- `GET /api/datasets/{id}/items`
- `POST /api/datasets/{id}/items`
- `POST /api/datasets/{id}/items/batch`
- `POST /api/datasets/{id}/items/delete`
- `POST /api/datasets/{id}/items/move`
- `POST /api/datasets/{id}/process`
- `GET /api/datasets/{id}/items/{split}/{item_id}`
- `GET /api/datasets/{id}/items/{split}/{item_id}/image`
- `PATCH /api/datasets/{id}/items/{split}/{item_id}/label`
- `PATCH /api/datasets/{id}/items/{split}/labels`
- `POST /api/datasets/{id}/items/{split}/{item_id}/preprocess-preview`
- `PUT /api/datasets/{id}/items/{split}/{item_id}/annotations`
- `GET /api/datasets/{id}/versions`
- `POST /api/datasets/{id}/versions`
- `GET /api/datasets/{id}/eda?split=...`
- `DELETE /api/datasets/{id}`
- `POST /api/datasets/{id}/labels`
- `PATCH /api/datasets/{id}/labels/{index}`
- `DELETE /api/datasets/{id}/labels/{index}`

## Behavior

- Reference YOLO and COCO datasets under `datasets/` are browsable and read-only.
- Editable datasets are stored under ignored `storage/datasets/{dataset_id}` with an `unassigned` inbox plus train/valid/test splits.
- Editable dataset manifests include `project_id`, `task_type`, `format`, and arbitrary labels.
- Editable dataset manifests may include an allowlisted preprocessing config with enabled flag, preset, resize, normalize, and safe transform names.
- Editable dataset manifests may include a split config with train/valid/test proportions, seed, stratify, and resplit-all flags.
- New uploads default to the `unassigned` inbox. Proceed saves config and splits inbox items into train/valid/test with default 70/20/10 proportions and seed `42`.
- Users can manually move selected items between unassigned/train/valid/test after processing.
- Preprocessing config includes random training-time augmentation or materialized version copies; materialized copies default to train split and are stored under ignored `storage/dataset_versions`.
- Editable image datasets support classification labels, object detection boxes, and segmentation polygons.
- Dataset name, metadata, and preprocessing config can be edited for local editable datasets; task type and format are immutable after creation.
- Classification uploads can attach a label immediately, and classification image labels can be changed later without drawing tools.
- Editable datasets support multi-image upload, selected item deletion, and bulk classification label edits.
- Preprocess previews write owned temporary artifacts under ignored `storage/previews` and do not modify source images.
- Version creation writes generated artifacts outside the source dataset and leaves original images unchanged.
- EDA summaries are computed on demand with split counts, class balance, unlabeled counts, image size/aspect ratio summaries, annotation totals, and warnings.
- YOLO-compatible datasets write `data.yaml` and `labels/*.txt` for training/evaluation compatibility.
- General annotations are persisted as JSON sidecars under split `annotations/` folders.
- Annotation writes are allowed only for editable datasets.
- Label deletion is blocked while annotations use the label unless forced.
- Editable dataset deletion hard-deletes the owned storage folder; reference deletion is blocked.
- Dataset images are served through API file routes so reference assets do not need to move into `storage/`.

## Acceptance Criteria

- Frontend opens to a catalog-first dataset list with split counts, status, and create actions.
- Clicking a dataset opens a workspace with Images, Annotate, and Config tabs; Images handles upload/browse/move/delete, Annotate handles labels and annotation editing, and Config handles preprocessing/splitting/version artifacts.
- Users can create editable classification, detection, and segmentation datasets.
- Users can upload classification images with labels and edit each image label after upload.
- Users can upload multiple images at once and delete or relabel selected images.
- Users can add labels and save task-aware annotations.
- Users can expand preprocessing and dataset split accordions, preview allowlisted preprocessing on a single item, create materialized version artifacts, and training uses prepared copies rather than changing originals.
- Users can use dataset options menus in both catalog cards and the dataset workspace to open, rename where applicable, duplicate, or delete editable datasets.
- EDA is available as a first-class dataset workspace tab with split counts, class balance, geometry, health, and warning summaries.
- Users can clone YOLO-compatible datasets.
- Users can delete editable datasets without modifying references.
- Original `datasets/`, `models/`, and `notebooks/` content is not rewritten.
