from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse

from app.container import dataset_service
from app.schemas import (
    DatasetAnnotationSave,
    DatasetCloneRequest,
    DatasetCreate,
    DatasetEdaSummary,
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
    DatasetPreprocessPreview,
    DatasetPreprocessPreviewRequest,
    DatasetProcessRequest,
    DatasetProcessResponse,
    DatasetSummary,
    DatasetUpdate,
    DatasetVersionCreate,
    DatasetVersionSummary,
    DeleteResponse,
)

router = APIRouter(prefix="/datasets")


@router.get("", response_model=list[DatasetSummary])
def list_datasets(project_id: str | None = Query(default=None)) -> list[DatasetSummary]:
    return dataset_service.list_datasets(project_id)


@router.post("", response_model=DatasetSummary)
def create_dataset(payload: DatasetCreate) -> DatasetSummary:
    return dataset_service.create_dataset(payload)


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
