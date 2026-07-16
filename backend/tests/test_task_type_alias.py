"""The `"text"` task type is a deprecated alias for `"text_classification"`.

These tests pin down the deprecation window described in
`specs/improvements/10-openapi-types.md`: submitting the legacy alias on
input must still be accepted and silently normalized to the canonical
value, while the canonical value continues to round-trip unchanged.
"""

import pytest

from app.schemas import (
    DatasetCreate,
    DatasetImportRequest,
    ProjectCreate,
    ProjectUpdate,
    TrainingJobCreate,
)

# Minimal required-field kwargs per model, so the parametrized tests below can
# focus purely on `task_type` normalization.
_REQUIRED_KWARGS = {
    DatasetCreate: {"name": "demo-dataset"},
    DatasetImportRequest: {"path": "/tmp/demo-dataset"},
    TrainingJobCreate: {},
}


@pytest.mark.parametrize(
    "model_cls",
    [DatasetCreate, DatasetImportRequest, TrainingJobCreate],
)
def test_legacy_text_alias_normalizes_to_canonical_task_type(model_cls) -> None:
    instance = model_cls(task_type="text", **_REQUIRED_KWARGS[model_cls])
    assert instance.task_type == "text_classification"


@pytest.mark.parametrize(
    "model_cls",
    [DatasetCreate, DatasetImportRequest, TrainingJobCreate],
)
def test_canonical_task_type_is_unaffected(model_cls) -> None:
    instance = model_cls(task_type="text_classification", **_REQUIRED_KWARGS[model_cls])
    assert instance.task_type == "text_classification"


def test_project_create_task_types_list_normalizes_legacy_alias() -> None:
    project = ProjectCreate(name="demo", task_types=["classification", "text"])
    assert project.task_types == ["classification", "text_classification"]


def test_project_update_task_types_list_normalizes_legacy_alias() -> None:
    update = ProjectUpdate(task_types=["text"])
    assert update.task_types == ["text_classification"]


def test_unknown_task_type_still_rejected() -> None:
    with pytest.raises(ValueError):
        DatasetCreate(name="demo-dataset", task_type="not_a_real_task_type")
