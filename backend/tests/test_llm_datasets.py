"""Phase 10: LLM fine-tuning records, formats, EDA, origin, and Hub import."""

import json
from io import BytesIO
from pathlib import Path

import anyio
import pytest
from fastapi import HTTPException, UploadFile

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import (
    DatasetCreate,
    DatasetHubColumnMapping,
    DatasetHubImportRequest,
    DatasetItemDeleteRequest,
    DatasetItemMoveRequest,
    DatasetProcessRequest,
    DatasetRecordCreate,
    DatasetRecordSave,
    DatasetSplitConfig,
)
from app.services.dataset_hub import DatasetHubService
from app.services.datasets import DatasetService


def make_service(settings: Settings) -> DatasetService:
    storage = Storage(settings)
    storage.ensure()
    return DatasetService(settings, storage)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_create_instruction_dataset_and_regenerate_jsonl(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="Alpaca SFT", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    assert dataset.task_type == "llm_finetune"
    assert dataset.format == "instruction_jsonl"
    assert dataset.origin == "created"
    assert dataset.labels == []

    detail = service.create_record(
        dataset.id,
        DatasetRecordCreate(split="train", record={"instruction": "Add 2+2", "output": "4"}),
    )
    assert detail.media_type == "record"
    assert detail.record == {"instruction": "Add 2+2", "output": "4"}
    assert detail.token_estimate > 0

    jsonl = read_jsonl(service.dataset_root(dataset.id) / "train" / "data.jsonl")
    assert jsonl == [{"instruction": "Add 2+2", "output": "4"}]


def test_instruction_record_requires_output(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="I", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    with pytest.raises(HTTPException) as exc:
        service.create_record(
            dataset.id, DatasetRecordCreate(split="train", record={"instruction": "hi", "output": "  "})
        )
    assert exc.value.status_code == 422


def test_qa_maps_into_instruction_shape(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="QA", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    detail = service.create_record(
        dataset.id,
        DatasetRecordCreate(
            split="train",
            record={"question": "Capital of France?", "context": "Geography", "answer": "Paris"},
        ),
    )
    assert detail.record == {"instruction": "Capital of France?", "output": "Paris", "input": "Geography"}


def test_chat_record_validation_and_excerpts(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="Chat", task_type="llm_finetune", format="chat_jsonl", labels=[])
    )
    detail = service.create_record(
        dataset.id,
        DatasetRecordCreate(
            split="train",
            record={
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            },
        ),
    )
    assert detail.text_preview == "Hi"
    assert detail.output_preview == "Hello"


def test_chat_record_requires_user_and_assistant(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="Chat2", task_type="llm_finetune", format="chat_jsonl", labels=[])
    )
    with pytest.raises(HTTPException) as exc:
        service.create_record(
            dataset.id,
            DatasetRecordCreate(split="train", record={"messages": [{"role": "user", "content": "Hi"}]}),
        )
    assert exc.value.status_code == 422


def test_sharegpt_conversations_normalize_to_messages(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="ShareGPT", task_type="llm_finetune", format="chat_jsonl", labels=[])
    )
    detail = service.create_record(
        dataset.id,
        DatasetRecordCreate(
            split="train",
            record={
                "conversations": [
                    {"from": "human", "value": "What is 2+2?"},
                    {"from": "gpt", "value": "4"},
                ]
            },
        ),
    )
    assert detail.record == {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }


def test_edit_record_rewrites_jsonl(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="Edit", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    detail = service.create_record(
        dataset.id, DatasetRecordCreate(split="train", record={"instruction": "a", "output": "b"})
    )
    service.save_record(
        dataset.id, "train", detail.id, DatasetRecordSave(record={"instruction": "a", "output": "c"})
    )
    jsonl = read_jsonl(service.dataset_root(dataset.id) / "train" / "data.jsonl")
    assert jsonl == [{"instruction": "a", "output": "c"}]


def test_delete_and_move_records_regenerate_jsonl(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="Move", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    first = service.create_record(
        dataset.id, DatasetRecordCreate(split="unassigned", record={"instruction": "one", "output": "1"})
    )
    service.create_record(
        dataset.id, DatasetRecordCreate(split="unassigned", record={"instruction": "two", "output": "2"})
    )
    root = service.dataset_root(dataset.id)
    assert len(read_jsonl(root / "unassigned" / "data.jsonl")) == 2

    service.move_items(
        dataset.id,
        DatasetItemMoveRequest(source_split="unassigned", target_split="train", ids=[first.id]),
    )
    assert len(read_jsonl(root / "unassigned" / "data.jsonl")) == 1
    assert len(read_jsonl(root / "train" / "data.jsonl")) == 1

    remaining = service.list_items(dataset.id, "unassigned")
    service.delete_items(dataset.id, DatasetItemDeleteRequest(split="unassigned", ids=[remaining[0].id]))
    assert read_jsonl(root / "unassigned" / "data.jsonl") == []


def test_split_llm_dataset_distributes_records(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="Split", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    for i in range(10):
        service.create_record(
            dataset.id, DatasetRecordCreate(split="unassigned", record={"instruction": f"q{i}", "output": f"a{i}"})
        )
    response = service.process_dataset(
        dataset.id,
        DatasetProcessRequest(split=DatasetSplitConfig(train=0.7, valid=0.2, test=0.1)),
    )
    assert sum(response.moved.values()) == 10
    root = service.dataset_root(dataset.id)
    assert len(read_jsonl(root / "unassigned" / "data.jsonl")) == 0
    assert len(read_jsonl(root / "train" / "data.jsonl")) == response.moved["train"]


def test_llm_eda_reports_lengths_and_warnings(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="Eda", task_type="llm_finetune", format="chat_jsonl", labels=[])
    )
    record = {
        "messages": [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "general kenobi"},
        ]
    }
    service.create_record(dataset.id, DatasetRecordCreate(split="train", record=record))
    service.create_record(dataset.id, DatasetRecordCreate(split="train", record=dict(record)))
    eda = service.eda_summary(dataset.id, "train")
    assert eda.item_count == 2
    assert eda.role_counts.get("assistant") == 2
    assert eda.duplicate_count == 1
    assert any("duplicate" in warning for warning in eda.warnings)
    assert eda.text_length["max_tokens"] is not None


def test_legacy_manifest_without_origin_defaults_to_created(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(DatasetCreate(name="Legacy"))
    manifest_path = service.dataset_root(dataset.id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("origin", None)
    manifest.pop("origin_ref", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert service.summary(dataset.id).origin == "created"


def test_upload_records_jsonl_file(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="Upload", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    content = (
        '{"instruction": "one", "output": "1"}\n'
        '{"instruction": "two", "input": "ctx", "output": "2"}\n'
        '{"instruction": "bad", "output": ""}\n'  # invalid → skipped
    )
    upload = UploadFile(file=BytesIO(content.encode("utf-8")), filename="records.jsonl")
    response = anyio.run(service.upload_records_file, dataset.id, "train", upload)
    assert response.imported == 2
    assert response.skipped == 1
    jsonl = read_jsonl(service.dataset_root(dataset.id) / "train" / "data.jsonl")
    assert {"instruction": "two", "output": "2", "input": "ctx"} in jsonl


def test_upload_records_rejects_unsupported_file(settings: Settings):
    service = make_service(settings)
    dataset = service.create_dataset(
        DatasetCreate(name="UploadBad", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    upload = UploadFile(file=BytesIO(b"hi"), filename="notes.txt")
    with pytest.raises(HTTPException) as exc:
        anyio.run(service.upload_records_file, dataset.id, "train", upload)
    assert exc.value.status_code == 400


def test_hub_import_maps_columns_and_marks_origin(settings: Settings, monkeypatch):
    service = make_service(settings)
    hub = DatasetHubService(settings, service.storage, service)

    monkeypatch.setattr(hub, "_configs_and_splits", lambda hub_id, config: (["default"], ["train"]))
    monkeypatch.setattr(hub, "_revision", lambda hub_id: "abc123")

    rows = [
        {"instruction": "Q1", "input": "", "output": "A1"},
        {"instruction": "Q2", "input": "ctx", "output": "A2"},
        {"instruction": "bad", "input": "", "output": ""},  # invalid → skipped
    ]

    def fake_fetch(hub_id, config, split, offset, length):
        return (rows if offset == 0 else [], list(rows[0].keys()))

    monkeypatch.setattr(hub, "_fetch_rows", fake_fetch)

    response = hub.import_hub(
        DatasetHubImportRequest(
            hub_id="tatsu-lab/alpaca",
            task_type="llm_finetune",
            format="instruction_jsonl",
            name="Imported Alpaca",
            mapping=DatasetHubColumnMapping(instruction="instruction", input="input", output="output"),
        )
    )
    assert response.imported_rows == 2
    assert response.skipped_rows == 1
    assert response.dataset.origin == "imported_hf"
    assert response.dataset.origin_ref == "tatsu-lab/alpaca@abc123"
    jsonl = read_jsonl(service.dataset_root(response.dataset.id) / "unassigned" / "data.jsonl")
    assert {"instruction": "Q2", "output": "A2", "input": "ctx"} in jsonl


def test_hub_import_rejects_incomplete_mapping(settings: Settings):
    service = make_service(settings)
    hub = DatasetHubService(settings, service.storage, service)
    with pytest.raises(HTTPException) as exc:
        hub.import_hub(
            DatasetHubImportRequest(
                hub_id="x/y",
                task_type="llm_finetune",
                format="chat_jsonl",
                mapping=DatasetHubColumnMapping(),
            )
        )
    assert exc.value.status_code == 422


def test_hub_import_atomic_leaves_no_partial_on_empty(settings: Settings, monkeypatch):
    service = make_service(settings)
    hub = DatasetHubService(settings, service.storage, service)
    monkeypatch.setattr(hub, "_configs_and_splits", lambda hub_id, config: (["default"], ["train"]))
    monkeypatch.setattr(hub, "_revision", lambda hub_id: "main")
    monkeypatch.setattr(hub, "_fetch_rows", lambda *a, **k: ([], []))
    with pytest.raises(HTTPException):
        hub.import_hub(
            DatasetHubImportRequest(
                hub_id="x/y",
                task_type="llm_finetune",
                format="instruction_jsonl",
                mapping=DatasetHubColumnMapping(instruction="instruction", output="output"),
            )
        )
    # No leftover temp dirs and no imported dataset in the catalog.
    leftovers = list(service.storage.datasets.glob(".import-*"))
    assert leftovers == []
    assert service.list_datasets() == [] or all(
        d.origin != "imported_hf" for d in service.list_datasets()
    )
