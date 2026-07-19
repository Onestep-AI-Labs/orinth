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

- Reference YOLO and COCO datasets under `datasets/` are browsable and read-only. They stay scoped to the default workspace: `datasets/` is git-ignored local data, so they cannot be relied on to exist elsewhere.
- Tracked starter samples under `sample_data/` are committed, so a fresh clone has usable data with no seeding step. They are marked `shared`, which makes them visible read-only in every project while belonging to none — `count_owned_datasets` excludes them, or a project owning nothing would be undeletable.
- `shared` is a field on the dataset, not an id-prefix convention. The prefix check it replaced was duplicated across the dataset and evaluation services and dropped a sample out of both the moment it was renamed.
- Shared samples are filtered to the requesting project's declared `task_types`, so an NLP-only workspace is never offered an image dataset. A project's own datasets are always listed regardless of task.
- Current samples: `sample_image_classification` (300 waste images across six classes, 210/60/30) and the three NLP samples. Object detection and segmentation have no tracked sample, so a project declaring only those tasks sees none.
- Editable datasets are stored under ignored `storage/datasets/{dataset_id}` with an `unassigned` inbox plus train/valid/test splits.
- Editable dataset manifests include `project_id`, `task_type`, `format`, and arbitrary labels.
- The dataset create panel asks for a class-label list only where labels are meaningful — image classification, text classification, object detection, and segmentation. Summarization and question answering carry their content in the annotation payload and have a fixed label (`summary` / `answer`), which the client sends regardless of the label draft. Offering the field there previously saved those datasets with a stray `object` label.
- The label list starts empty and Create dataset stays disabled until at least one label is entered. A prefilled placeholder label shipped on every dataset whose author did not notice it. Labels are entered one at a time with Enter, or several at once separated by `;` or `,`.
- Storage format is derived from the task, never chosen, so it is shown as a caption on the task step rather than as its own field.
- Editable dataset manifests may include an allowlisted preprocessing config with enabled flag, preset, resize, normalize, and safe transform names.
- Editable dataset manifests may include a split config with train/valid/test proportions, seed, stratify, and resplit-all flags.
- New uploads default to the `unassigned` inbox. Proceed saves config and splits inbox items into train/valid/test with default 70/20/10 proportions and seed `42`.
- Users can manually move selected items between unassigned/train/valid/test after processing.
- Preprocessing config includes random training-time augmentation or materialized version copies; materialized copies default to train split and are stored under ignored `storage/dataset_versions`.
- Editable image datasets support classification labels, object detection boxes, and segmentation polygons.
- The annotation editor provides explicit select/move, bounding-box, and polygon tools; saved boxes and polygons can be selected and moved before saving.
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
- Users can switch annotation tools, draw boxes or polygons, select existing shapes, and move them within image bounds.
- Users can expand preprocessing and dataset split accordions, preview allowlisted preprocessing on a single item, create materialized version artifacts, and training uses prepared copies rather than changing originals.
- Users can use dataset options menus in both catalog cards and the dataset workspace to open, rename where applicable, duplicate, or delete editable datasets.
- EDA is available as a first-class dataset workspace tab with split counts, class balance, geometry, health, and warning summaries.
- Users can clone YOLO-compatible datasets.
- Users can delete editable datasets without modifying references.
- Original `datasets/`, `models/`, and `notebooks/` content is not rewritten.
