from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.container import dataset_hub_service, dataset_prep_service, dataset_service
from app.core.database import get_db
from app.core.defaults import DEFAULT_PROJECT_ID
from app.db.models import Project
from app.schemas import (
    DatasetAnnotationSave,
    DatasetCloneRequest,
    DatasetCreate,
    DatasetDetection,
    DatasetEdaSummary,
    DatasetHubFacets,
    DatasetHubImportRequest,
    DatasetHubImportResponse,
    DatasetHubPreview,
    DatasetHubSearchResponse,
    DatasetImportRequest,
    DatasetItemBatchUploadResponse,
    DatasetItemBulkLabelUpdate,
    DatasetItemBulkLabelUpdateResponse,
    DatasetItemDeleteRequest,
    DatasetItemDetail,
    DatasetItemLabelUpdate,
    DatasetItemMoveRequest,
    DatasetItemMoveResponse,
    DatasetItemPage,
    DatasetLabelCreate,
    DatasetLabelUpdate,
    DatasetPrepApplyRequest,
    DatasetPrepPlan,
    DatasetPrepResponse,
    DatasetPreprocessPreview,
    DatasetPreprocessPreviewRequest,
    DatasetPrepStartRequest,
    DatasetPrepStatus,
    DatasetProcessRequest,
    DatasetProcessResponse,
    DatasetReadiness,
    DatasetRecordCreate,
    DatasetRecordSave,
    DatasetRecordUploadResponse,
    DatasetSummary,
    DatasetTablePage,
    DatasetUpdate,
    DatasetVersionCreate,
    DatasetVersionSummary,
    DeleteResponse,
)
from app.services import hub_facets
from app.services.datasets.prep.apply import PrepApplyError
from app.services.datasets.prep.service import PrepBusyError

router = APIRouter(prefix="/datasets")


def _require_project_task(db: Session, project_id: str, task_type: str) -> None:
    """Block create/import of a task the project has not declared (phase 10).

    Mirrors the `llm_finetune` gating the create panel and hub import enforce in
    the UI; the backend is the authority so a direct API call cannot bypass it.
    """
    project = db.get(Project, project_id)
    if project is None:
        return
    declared = list(project.task_types or [])
    if declared and task_type not in declared:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Project does not allow '{task_type}' datasets. Add the task in project settings first."
            ),
        )


# ---- prep agent (phase 21) ------------------------------------------------
#
# Registered ahead of every `/{dataset_id}` route below: FastAPI matches in
# declaration order, so a literal path added later would be swallowed by the
# path parameter and answer 404 for `/datasets/ingest`. The `/hub/*` routes
# above take the same care.


@router.post("/ingest", response_model=DatasetSummary)
def ingest_dataset(
    project_id: str = Form(DEFAULT_PROJECT_ID),
    name: str | None = Form(None),
    files: list[UploadFile] = File(...),
    relative_paths: list[str] | None = Form(None),
) -> DatasetSummary:
    """Create a dataset from raw files, with nothing declared about them.

    No task type, no format, no labels — that is the point. `relative_paths`
    carries each file's path inside the folder the user dropped, because the
    directory layout is what detection reads; without it every upload looks like
    a flat pile of files.
    """
    dataset_id = dataset_prep_service.create_draft(project_id=project_id, name=name)
    _stage_uploads(dataset_id, files, relative_paths)
    return dataset_service.summary(dataset_id)


@router.post("/{dataset_id}/ingest", response_model=DatasetSummary)
def ingest_into_dataset(
    dataset_id: str,
    files: list[UploadFile] = File(...),
    relative_paths: list[str] | None = Form(None),
) -> DatasetSummary:
    _stage_uploads(dataset_id, files, relative_paths)
    return dataset_service.summary(dataset_id)


def _stage_uploads(
    dataset_id: str, files: list[UploadFile], relative_paths: list[str] | None
) -> None:
    paths = list(relative_paths or [])
    skipped = 0
    for index, upload in enumerate(files):
        relative = paths[index] if index < len(paths) else None
        stored = dataset_prep_service.stage_file(
            dataset_id,
            stream=upload.file,
            filename=upload.filename or f"file-{index}",
            relative_path=relative,
        )
        if not stored:
            skipped += 1
    if files and skipped == len(files):
        raise HTTPException(
            status_code=422,
            detail="None of the uploaded files could be stored. Check their names and sizes.",
        )


@router.get("/{dataset_id}/prep/detect", response_model=DatasetDetection)
def detect_dataset(dataset_id: str) -> DatasetDetection:
    """What the deterministic scan sees, with no plan and no model call."""
    return DatasetDetection.model_validate(
        dataset_prep_service.detect_dataset(dataset_id).__dict__
    )


@router.post("/{dataset_id}/prep", response_model=DatasetSummary, status_code=202)
def run_dataset_prep(
    dataset_id: str,
    payload: DatasetPrepStartRequest | None = None,
    db: Session = Depends(get_db),
) -> DatasetSummary:
    """Start a prep run: detect, plan, and (by default) apply.

    Returns as soon as the run is queued, with the dataset already reading
    `prep.state == "planning"`. Applying a plan to a large upload is tens of
    thousands of file operations — 37 seconds on a real 15,000-row table — which
    is longer than a dev proxy will hold a request open, and it competes with
    the studio's own polling of the same dataset. The client polls
    `GET /datasets/{id}` (or `/prep` for the plan) until the state settles.

    The project's declared task types gate the plan, so the agent never proposes
    a task that apply would reject with a 409.
    """
    request = payload or DatasetPrepStartRequest()
    summary = dataset_service.summary(dataset_id)
    allowed = _project_task_types(db, summary.project_id)

    try:
        return dataset_prep_service.start(
            dataset_id, allowed_task_types=allowed, auto_apply=request.auto_apply
        )
    except PrepBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PrepApplyError as error:
        # Only reachable on the inline path (no executor configured).
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/{dataset_id}/prep/status", response_model=DatasetPrepStatus)
def get_dataset_prep_status(dataset_id: str) -> DatasetPrepStatus:
    """The run state alone, cheap enough to poll while a run is in flight.

    Declared before `/{dataset_id}/prep` so the literal segment wins the match.
    """
    return dataset_prep_service.prep_status(dataset_id)


@router.get("/{dataset_id}/prep", response_model=DatasetPrepPlan)
def get_dataset_prep(dataset_id: str) -> DatasetPrepPlan:
    plan = dataset_prep_service.stored_plan(dataset_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="No prep run for this dataset yet")
    return plan


@router.post("/{dataset_id}/prep/apply", response_model=DatasetPrepResponse)
def apply_dataset_prep(
    dataset_id: str, payload: DatasetPrepApplyRequest, db: Session = Depends(get_db)
) -> DatasetPrepResponse:
    """Apply a plan the user may have edited.

    The project gate lands here rather than at ingest: this is the moment the
    dataset claims a task, which is what the gate is about.
    """
    summary = dataset_service.summary(dataset_id)
    if payload.plan.task_type:
        _require_project_task(db, summary.project_id, str(payload.plan.task_type))
    try:
        dataset = dataset_prep_service.apply(dataset_id, payload.plan)
    except PrepApplyError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return DatasetPrepResponse(dataset=dataset, plan=payload.plan)


@router.post("/{dataset_id}/prep/undo", response_model=DatasetSummary)
def undo_dataset_prep(dataset_id: str) -> DatasetSummary:
    try:
        return dataset_prep_service.undo(dataset_id)
    except PrepApplyError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/{dataset_id}/prep/discard-staged", response_model=DatasetSummary)
def discard_staged_files(dataset_id: str) -> DatasetSummary:
    """Reclaim the raw upload once the prepared dataset looks right."""
    return dataset_prep_service.discard_staged(dataset_id)


def _project_task_types(db: Session, project_id: str) -> list[str] | None:
    project = db.get(Project, project_id)
    if project is None:
        return None
    declared = list(project.task_types or [])
    return declared or None


@router.get("", response_model=list[DatasetSummary])
def list_datasets(
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[DatasetSummary]:
    # The shared starter samples are filtered to the project's own task types,
    # so a vision workspace is not offered text datasets and vice versa.
    task_types = None
    if project_id:
        project = db.get(Project, project_id)
        if project is not None:
            task_types = list(project.task_types or [])
    return dataset_service.list_datasets(project_id, task_types)


@router.post("", response_model=DatasetSummary)
def create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)) -> DatasetSummary:
    _require_project_task(db, payload.project_id, payload.task_type)
    return dataset_service.create_dataset(payload)


@router.get("/hub/facets", response_model=DatasetHubFacets)
def get_hub_facets() -> DatasetHubFacets:
    """The browse vocabulary and what each term means.

    Served rather than hardcoded in the client so the tooltip on a filter chip
    and the tooltip on the badge showing the same term on a result card come
    from one string. See `services/hub_facets.py`.
    """
    return hub_facets.facets()


@router.get("/hub/search", response_model=DatasetHubSearchResponse)
def search_hub_datasets(
    query: str | None = Query(default=None),
    task: str | None = Query(default=None),
    limit: int = Query(default=24, ge=1, le=100),
    modality: list[str] = Query(default_factory=list),
    format: list[str] = Query(default_factory=list),
    size: list[str] = Query(default_factory=list),
    task_category: list[str] = Query(default_factory=list),
    sort: str = Query(default="trending"),
) -> DatasetHubSearchResponse:
    """Browse the Hub with the same facets huggingface.co exposes.

    The repeated-key parameters (`?modality=text&modality=tabular`) mirror the
    Hub's own query shape, so a filter set here is transferable to a URL there.
    """
    return dataset_hub_service.search(
        query,
        task,
        limit,
        modalities=modality,
        formats=format,
        sizes=size,
        tasks=task_category,
        sort=sort,
    )


@router.get("/hub/preview", response_model=DatasetHubPreview)
def preview_hub_dataset(
    hub_id: str = Query(...),
    config: str | None = Query(default=None),
    split: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=100),
) -> DatasetHubPreview:
    return dataset_hub_service.preview(hub_id, config, split, limit)


@router.post("/import/hub", response_model=DatasetHubImportResponse)
def import_hub_dataset(
    payload: DatasetHubImportRequest, db: Session = Depends(get_db)
) -> DatasetHubImportResponse:
    _require_project_task(db, payload.project_id, payload.task_type)
    return dataset_hub_service.import_hub(payload)


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_dataset(dataset_id: str) -> DatasetSummary:
    """One dataset, without building the whole catalog to find it.

    The studio gets a single dataset by fetching `GET /datasets` and filtering,
    which is fine for a page that wants the catalog anyway and wrong for anything
    that wants one: `_split_summary` walks items and reads annotation JSON per
    item, *for every dataset*, which phase 21 measured at seconds on a real
    workspace. `orinth dataset show` and every post-job readiness re-read go
    through here instead.

    **Declaration order is load-bearing.** This must stay below `/ingest`,
    `/hub/*`, and `/import/*`, or the path parameter swallows those literals.
    """
    return dataset_service.summary(dataset_id)


@router.get("/{dataset_id}/download")
def download_dataset(
    dataset_id: str,
    splits: str | None = Query(default=None, description="Comma-separated subset"),
) -> FileResponse:
    """The dataset as a zip.

    Cannot be built from what already exists: `POST /{id}/versions` writes a
    snapshot to a server-side path and nothing streams it back. This mirrors
    `GET /api/models/{model_id}/download`, which already bundles a multi-asset
    model the same way, and gives the dataset catalog the download action the
    model catalog has had since phase 12.
    """
    wanted = [part.strip() for part in (splits or "").split(",") if part.strip()]
    archive = dataset_service.archive_dataset(dataset_id, splits=wanted or None)
    return FileResponse(archive, media_type="application/zip", filename=archive.name)


@router.patch("/{dataset_id}", response_model=DatasetSummary)
def update_dataset(dataset_id: str, payload: DatasetUpdate) -> DatasetSummary:
    return dataset_service.update_dataset(dataset_id, payload)


@router.post("/{dataset_id}/process", response_model=DatasetProcessResponse)
def process_dataset(dataset_id: str, payload: DatasetProcessRequest) -> DatasetProcessResponse:
    return dataset_service.process_dataset(dataset_id, payload)


@router.post("/import", response_model=DatasetSummary)
def import_dataset(payload: DatasetImportRequest) -> DatasetSummary:
    return dataset_service.import_dataset(payload)


@router.post("/{dataset_id}/clone", response_model=DatasetSummary)
def clone_dataset(dataset_id: str, payload: DatasetCloneRequest) -> DatasetSummary:
    return dataset_service.clone_dataset(dataset_id, payload.name, payload.project_id)


@router.delete("/{dataset_id}", response_model=DeleteResponse)
def delete_dataset(dataset_id: str) -> DeleteResponse:
    dataset_service.delete_dataset(dataset_id)
    return DeleteResponse(deleted=1)


@router.post("/{dataset_id}/labels", response_model=DatasetSummary)
def add_dataset_label(dataset_id: str, payload: DatasetLabelCreate) -> DatasetSummary:
    return dataset_service.add_label(dataset_id, payload.name)


@router.patch("/{dataset_id}/labels/{label_index}", response_model=DatasetSummary)
def rename_dataset_label(
    dataset_id: str, label_index: int, payload: DatasetLabelUpdate
) -> DatasetSummary:
    return dataset_service.rename_label(dataset_id, label_index, payload.name)


@router.delete("/{dataset_id}/labels/{label_index}", response_model=DatasetSummary)
def delete_dataset_label(
    dataset_id: str,
    label_index: int,
    force: bool = Query(default=False),
) -> DatasetSummary:
    return dataset_service.delete_label(dataset_id, label_index, force)


@router.get("/{dataset_id}/items", response_model=DatasetItemPage)
def list_dataset_items(
    dataset_id: str,
    split: str = Query("train"),
    class_name: str | None = Query(default=None),
    unlabeled: bool = Query(default=False),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> DatasetItemPage:
    return dataset_service.list_items_page(dataset_id, split, class_name, unlabeled, limit, offset)


@router.get("/{dataset_id}/table", response_model=DatasetTablePage)
def get_dataset_table(
    dataset_id: str,
    split: str = Query("all"),
    class_name: str | None = Query(default=None),
    unlabeled: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> DatasetTablePage:
    """The dataset as a grid: same rows as `/items`, projected into columns.

    Deliberately built on `list_items_page` rather than on a materialized index.
    That read already decides the page from the file paths and opens only the
    window, so a page here costs what a page of the image browser costs — and
    there is no second copy of the data that can drift from what the training
    runners actually load. See `services/datasets/table.py`.
    """
    return dataset_service.table_page(
        dataset_id, split, class_name, unlabeled, limit, offset
    )


@router.post("/{dataset_id}/items", response_model=DatasetItemDetail)
async def upload_dataset_item(
    dataset_id: str,
    split: str = Form("unassigned"),
    class_id: int | None = Form(default=None),
    class_name: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> DatasetItemDetail:
    return await dataset_service.upload_image(dataset_id, split, file, class_id, class_name)


@router.post("/{dataset_id}/items/batch", response_model=DatasetItemBatchUploadResponse)
async def upload_dataset_items_batch(
    dataset_id: str,
    split: str = Form("unassigned"),
    class_id: int | None = Form(default=None),
    class_name: str | None = Form(default=None),
    files: list[UploadFile] = File(...),
) -> DatasetItemBatchUploadResponse:
    return await dataset_service.upload_images(dataset_id, split, files, class_id, class_name)


@router.post("/{dataset_id}/records", response_model=DatasetItemDetail)
def create_dataset_record(dataset_id: str, payload: DatasetRecordCreate) -> DatasetItemDetail:
    return dataset_service.create_record(dataset_id, payload)


@router.post("/{dataset_id}/records/upload", response_model=DatasetRecordUploadResponse)
async def upload_dataset_records(
    dataset_id: str,
    split: str = Form("unassigned"),
    file: UploadFile = File(...),
) -> DatasetRecordUploadResponse:
    return await dataset_service.upload_records_file(dataset_id, split, file)


@router.put("/{dataset_id}/records/{split}/{item_id}", response_model=DatasetItemDetail)
def save_dataset_record(
    dataset_id: str, split: str, item_id: str, payload: DatasetRecordSave
) -> DatasetItemDetail:
    return dataset_service.save_record(dataset_id, split, item_id, payload)


@router.post("/{dataset_id}/items/delete", response_model=DeleteResponse)
def delete_dataset_items(dataset_id: str, payload: DatasetItemDeleteRequest) -> DeleteResponse:
    return dataset_service.delete_items(dataset_id, payload)


@router.post("/{dataset_id}/items/move", response_model=DatasetItemMoveResponse)
def move_dataset_items(
    dataset_id: str,
    payload: DatasetItemMoveRequest,
) -> DatasetItemMoveResponse:
    return dataset_service.move_items(dataset_id, payload)


@router.patch(
    "/{dataset_id}/items/{split}/labels",
    response_model=DatasetItemBulkLabelUpdateResponse,
)
def update_dataset_item_labels(
    dataset_id: str,
    split: str,
    payload: DatasetItemBulkLabelUpdate,
) -> DatasetItemBulkLabelUpdateResponse:
    return dataset_service.bulk_set_item_labels(dataset_id, split, payload)


@router.get("/{dataset_id}/items/{split}/{item_id}", response_model=DatasetItemDetail)
def get_dataset_item(dataset_id: str, split: str, item_id: str) -> DatasetItemDetail:
    return dataset_service.item_detail(dataset_id, split, item_id)


@router.get("/{dataset_id}/items/{split}/{item_id}/image")
def get_dataset_image(dataset_id: str, split: str, item_id: str) -> FileResponse:
    return FileResponse(dataset_service.image_path(dataset_id, split, item_id))


@router.get("/{dataset_id}/items/{split}/{item_id}/text")
def get_dataset_text(dataset_id: str, split: str, item_id: str) -> FileResponse:
    return FileResponse(dataset_service.text_path(dataset_id, split, item_id), media_type="text/plain")


@router.put(
    "/{dataset_id}/items/{split}/{item_id}/annotations",
    response_model=DatasetItemDetail,
)
def save_dataset_annotations(
    dataset_id: str,
    split: str,
    item_id: str,
    payload: DatasetAnnotationSave,
) -> DatasetItemDetail:
    return dataset_service.save_annotations(dataset_id, split, item_id, payload)


@router.patch(
    "/{dataset_id}/items/{split}/{item_id}/label",
    response_model=DatasetItemDetail,
)
def update_dataset_item_label(
    dataset_id: str,
    split: str,
    item_id: str,
    payload: DatasetItemLabelUpdate,
) -> DatasetItemDetail:
    return dataset_service.set_item_label(dataset_id, split, item_id, payload)


@router.post(
    "/{dataset_id}/items/{split}/{item_id}/preprocess-preview",
    response_model=DatasetPreprocessPreview,
)
def preview_dataset_preprocess(
    dataset_id: str,
    split: str,
    item_id: str,
    payload: DatasetPreprocessPreviewRequest,
) -> DatasetPreprocessPreview:
    return dataset_service.preprocess_preview(dataset_id, split, item_id, payload.config)


@router.get("/{dataset_id}/versions", response_model=list[DatasetVersionSummary])
def list_dataset_versions(dataset_id: str) -> list[DatasetVersionSummary]:
    return dataset_service.list_versions(dataset_id)


@router.post("/{dataset_id}/versions", response_model=DatasetVersionSummary)
def create_dataset_version(
    dataset_id: str,
    payload: DatasetVersionCreate,
) -> DatasetVersionSummary:
    return dataset_service.create_version(dataset_id, payload)


@router.get("/{dataset_id}/eda", response_model=DatasetEdaSummary)
def dataset_eda(dataset_id: str, split: str = Query("train")) -> DatasetEdaSummary:
    return dataset_service.eda_summary(dataset_id, split)


@router.get("/{dataset_id}/readiness", response_model=DatasetReadiness)
def dataset_readiness(dataset_id: str) -> DatasetReadiness:
    """Readiness alone, for polling after a mutation.

    The same value rides on every `DatasetSummary`; this route exists so a client
    that only wants to know whether the Train button should light up does not
    have to refetch the whole catalog to find out.
    """
    return dataset_service.summary(dataset_id).readiness
