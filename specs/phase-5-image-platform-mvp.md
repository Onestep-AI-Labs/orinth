# Spec: Phase 5 Image Platform MVP

## Status

Implemented MVP

## Goal

Generalize the local dental segmentation workspace into a project-scoped image data platform for:

- image classification,
- object detection,
- segmentation.

Text data is reserved as future metadata only. The platform remains a local research/engineering tool and must not imply autonomous clinical diagnosis.

## Interfaces

- `GET /api/projects`
- `POST /api/projects`
- `PATCH /api/projects/{project_id}`
- `DELETE /api/projects/{project_id}`
- `GET /api/datasets?project_id=...`
- `POST /api/datasets`
- `PATCH /api/datasets/{id}`
- `DELETE /api/datasets/{id}`
- `POST /api/datasets/{id}/items/batch`
- `POST /api/datasets/{id}/items/delete`
- `PATCH /api/datasets/{id}/items/{split}/labels`
- `GET /api/datasets/{id}/versions`
- `POST /api/datasets/{id}/versions`
- `GET /api/datasets/{id}/eda?split=...`
- `POST /api/datasets/{id}/process`
- `POST /api/datasets/{id}/items/move`
- `POST /api/datasets/{id}/labels`
- `PATCH /api/datasets/{id}/labels/{index}`
- `DELETE /api/datasets/{id}/labels/{index}?force=false`
- `PATCH /api/datasets/{id}/items/{split}/{item_id}/label`
- `POST /api/datasets/{id}/items/{split}/{item_id}/preprocess-preview`
- `POST /api/inference/delete`
- `DELETE /api/inference`
- `DELETE /api/inference/{id}`
- `POST /api/testing/jobs/delete`
- `DELETE /api/testing/jobs`
- `DELETE /api/testing/jobs/{id}`
- `POST /api/testing/jobs/batch`
- `GET /api/testing/jobs/{id}/comparison`
- `GET /api/training/model-options?task_type=...`
- `POST /api/training/model-assets/prepare`
- `POST /api/training/jobs/delete`
- `DELETE /api/training/jobs`
- `DELETE /api/training/jobs/{id}`

## Behavior

- A default research image project is created automatically for existing data.
- Project creation happens on a dedicated create page where users provide a name, a required short UI description, and one or more image task types.
- Project cards open workspaces and expose a three-dot menu for deletion; projects cannot be created inline from the project list.
- User-created projects can be removed when they do not own datasets or history; the default project cannot be deleted.
- Dataset creation follows the active project's allowed image task types.
- Editable dataset manifests include `project_id`, `task_type`, `format`, and `labels`.
- Editable dataset manifests can persist allowlisted preprocessing config for training-time prepared copies.
- Preprocessing config supports random training-time augmentation and materialized generated copies; generated version artifacts live under ignored `storage/dataset_versions`.
- Editable datasets support classification label-only annotations, detection boxes, and segmentation polygons.
- Dataset Studio is catalog-first, then opens a tabbed workspace: Images for upload/browse/move/delete, Annotate for labels and annotation editing, EDA for dataset health and distribution, and Config for preprocessing, split, and version artifacts.
- The root page lists projects outside the project workspace; selecting a project opens the project sidebar at Dataset Studio.
- Global navigation includes Projects and Settings only; project creation is launched from the project list. Project pages render global navigation first, then a collapsible project sidebar with project switching and Datasets, Models, Inference, Testing, and Training links.
- Sidebar navigation includes an Available Models section inside each project.
- Dataset Studio uses an unassigned inbox for new uploads, a Config tab for preprocessing/augmentation/split settings, and a 70/20/10 Proceed workflow.
- Classification datasets support upload-time labels, per-image label edits, and bulk selected-image relabeling.
- Reference datasets remain read-only and are not deleted or rewritten.
- Inference, testing, and training lists can be filtered by project.
- Inference model selection lists only available local/promoted trained models.
- Completed runnable training jobs are automatically copied into stable trained-model storage and registered for inference/testing.
- Testing can create comparison runs by selecting multiple available models for the same dataset.
- Testing comparison runs are grouped by `comparison_id`, and opening any job in the group shows all sibling model results.
- History deletion hard-deletes DB rows and owned `storage/` artifacts.
- Active testing/training jobs cannot be deleted until terminal.
- YOLO training remains runnable with dataset-driven class names.
- Training selection is task-first; model options are filtered by classification, object detection, or segmentation.
- The training UI defaults to classification and selects the first runnable classification option when available.
- Ultralytics YOLO11 and YOLO26 detection/segmentation options are runnable through the YOLO family runner.
- Ultralytics SAM3, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, and YOLO-World are listed but gated until validated.
- Keras Applications classification training is runnable through a subprocess runner.
- Keras Applications model options include official CNN families such as MobileNetV2, EfficientNet B0-B7, EfficientNetV2, ResNet50, Xception, InceptionV3, DenseNet121, and ConvNeXtTiny, with constructor-style defaults based on the Keras Applications API.
- Hugging Face/transformer models are cataloged but gated until validated.
- External Keras asset preparation is explicit and stores markers under ignored `storage/model_assets`.
- Frontend navigation uses a global sidebar outside projects, a global rail plus separate collapsible project sidebar inside selected projects, and separate detail pages for testing/training jobs.
- The frontend product identity is Onestep Vision with the tagline "One workspace for image intelligence" and supporting parent-brand language for ONESTEP.
- The global sidebar brand mark/name links to the project list at `/` in both full and compact navigation states.
- Dataset workspace selection is URL-backed through `/datasets?dataset={dataset_id}`; the plain `/datasets` route always shows the dataset catalog.
- Data tables remain semantic tables and are contained in responsive horizontal scroll wrappers so narrow viewports do not overflow the page.
- Testing and training workspaces use responsive one-column tablet layouts and card-style job rows on phone-sized viewports.
- Project creation uses the Onestep Vision brand treatment and richer task-type cards while preserving the existing project creation payload.
- Training detail pages show history charts, ROC/AUC for classification where available, and runner curve artifacts.
- Frontend data-loading pages use skeleton states for initial fetches and compact inline loading states for small refetches.
- Inference forms render parameters that match the selected model task.
- Testing dataset selection lists only reference test splits and editable dataset test splits.
- Testing detail pages include task-aware summary metrics, task-relevant confusion matrices, object summaries, and per-image detail tables without extra chart panels.
- Project, dataset, model, and reporting chips use compact soft-color tags sized for dense operational scanning.
- Model catalog cards use compact recipe-style tinted cards with task/source metadata, availability tags, class tags, and pinned actions.
- Dataset catalog cards use the same simple recipe-style card treatment as model cards, avoiding square metric blocks and overlapping tags.
- Training progress displays epoch processed/total when epoch metrics are available.
- Frontend API and media requests default to same-origin `/api` and `/media`, with Next.js proxying those routes to the local FastAPI backend for single-URL tunnel sharing.

## Acceptance Criteria

- Users can create a project, create an image dataset, add labels, upload images, and save task-aware annotations.
- Users can create a project from `/projects/new`, enter a required short description, choose multiple image task types, delete eligible projects from a project card menu, and switch active projects inside the project workspace.
- Users can recognize the app as Onestep Vision, click the global logo/name to return home, and use the branded create-project experience without changing project creation behavior.
- Users can manage datasets from a catalog-first view, upload multiple images, edit classification labels per image or in bulk, preview preprocessing safely, generate version artifacts, and inspect core EDA.
- Users can open a dataset via `/datasets?dataset={dataset_id}` and return to the catalog by clicking the Datasets sidebar link or the Back to catalog action.
- Table-heavy pages remain usable on mobile and small desktop widths without causing page-level horizontal overflow.
- Testing and training pages remain usable across desktop, tablet, and mobile widths, including run configuration forms, history actions, job rows, comparisons, and detail metrics.
- Users can delete editable datasets, selected history rows, and clear terminal history.
- Users can view testing comparisons and training details on separate pages.
- Users can inspect EDA as a dedicated dataset tab and see skeleton loading states across data-heavy pages.
- Users can choose task-compatible YOLO, Ultralytics YOLO11/YOLO26, or Keras classification training options, while unvalidated transformer/specialist options show as gated.
- Existing dental reference data remains available under the default project.
- The frontend can be shared through one public tunnel URL while API and media requests continue to resolve through the local backend proxy.
- Backend tests pass and frontend typecheck/lint/build pass.
