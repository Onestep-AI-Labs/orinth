"""LLM fine-tuning record storage, shape validation, and LLM EDA.

Records for the ``llm_finetune`` task live as one JSON file per item under
``<split>/records/<item_id>.json`` so the existing item move/delete routes work
unchanged. On every write the canonical ``<split>/data.jsonl`` is regenerated —
one record per line — mirroring the YOLO precedent of writing a flat training
input alongside the editable per-item files.

This is a mixin consumed by ``DatasetService`` (see ``service.py``); it relies on
core helpers (``_location``, ``_editable_location``, ``_is_llm_task``,
``_media_dir``, ``_validate_split``, ``summary``, ``storage``) provided by the
other mixins composed onto the facade class.
"""


import json
import re
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.storage import Storage
from app.schemas import (
    DatasetEdaSummary,
    DatasetItemDetail,
    DatasetRecordCreate,
    DatasetRecordSave,
    DatasetRecordUploadResponse,
)
from app.services.datasets.constants import CHAT_ROLES, RECORD_SUFFIX, SPLITS
from app.services.datasets.types import DatasetLocation

# ShareGPT ``from`` speaker → our chat role. A native re-implementation of the
# common community convention (see phase 10); no code is copied from Unsloth.
_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "system": "system",
}

_PREVIEW_LIMIT = 160
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class RecordsMixin:
    """CRUD + validation for ``llm_finetune`` records and their LLM EDA."""

    storage: Storage

    # ---- public CRUD --------------------------------------------------

    def create_record(self, dataset_id: str, payload: DatasetRecordCreate) -> DatasetItemDetail:
        location = self._editable_location(dataset_id)
        self._require_llm(location)
        self._validate_split(payload.split)
        record = self._validate_record(location, payload.record)
        path = self._write_record(location, payload.split, record)
        self._regenerate_records_jsonl(location, payload.split)
        return self._record_item_from_path(location, payload.split, path, include=True)

    def save_record(
        self, dataset_id: str, split: str, item_id: str, payload: DatasetRecordSave
    ) -> DatasetItemDetail:
        location = self._editable_location(dataset_id)
        self._require_llm(location)
        self._validate_split(split)
        path = self._record_path(location, split, item_id)
        record = self._validate_record(location, payload.record)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        self._regenerate_records_jsonl(location, split)
        return self._record_item_from_path(location, split, path, include=True)

    async def upload_records_file(
        self, dataset_id: str, split: str, file: UploadFile
    ) -> DatasetRecordUploadResponse:
        """Bulk-load records from a user's own ``.jsonl``/``.json``/``.csv`` file.

        Each row is validated against the dataset's record shape; malformed or
        unmappable rows are skipped and counted rather than failing the upload.
        """
        location = self._editable_location(dataset_id)
        self._require_llm(location)
        self._validate_split(split)
        suffix = Path(file.filename or "records.jsonl").suffix.lower() or ".jsonl"
        if suffix not in {".jsonl", ".json", ".csv"}:
            raise HTTPException(status_code=400, detail="Upload a .jsonl, .json, or .csv file of records")
        raw = (await file.read()).decode("utf-8", errors="replace")
        rows = self._records_from_upload(raw, suffix)
        if not rows:
            raise HTTPException(status_code=400, detail="No records found in the uploaded file")
        imported = 0
        skipped = 0
        for row in rows:
            try:
                record = self._validate_record(location, row)
            except HTTPException:
                skipped += 1
                continue
            except Exception:  # noqa: BLE001 — one malformed row must not fail the upload.
                skipped += 1
                continue
            self._write_record(location, split, record)
            imported += 1
        self._regenerate_records_jsonl(location, split)
        warnings = [f"{skipped} rows skipped (invalid or unmappable)"] if skipped else []
        return DatasetRecordUploadResponse(imported=imported, skipped=skipped, warnings=warnings)

    def _records_from_upload(self, raw: str, suffix: str) -> list[dict]:
        if suffix == ".json":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return []
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
            return [data] if isinstance(data, dict) else []
        # `.jsonl` rows are whole records; `.csv` rows are flat instruction columns.
        return self._text_upload_rows(raw, ".csv" if suffix == ".csv" else ".jsonl")

    # ---- storage helpers ----------------------------------------------

    def _record_paths(self, location: DatasetLocation, split: str) -> list[Path]:
        record_dir = location.root / split / "records"
        if not record_dir.exists():
            return []
        return sorted(path for path in record_dir.iterdir() if path.suffix.lower() == RECORD_SUFFIX)

    def _record_path(self, location: DatasetLocation, split: str, item_id: str) -> Path:
        filename = Path(item_id).name
        for path in self._record_paths(location, split):
            if path.name == filename:
                return path
        raise HTTPException(status_code=404, detail="Dataset record not found")

    def _write_record(self, location: DatasetLocation, split: str, record: dict) -> Path:
        record_dir = location.root / split / "records"
        record_dir.mkdir(parents=True, exist_ok=True)
        path = record_dir / f"rec-{uuid4().hex[:12]}{RECORD_SUFFIX}"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _read_record(self, path: Path) -> dict | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    def _regenerate_records_jsonl(self, location: DatasetLocation, split: str) -> None:
        """Rewrite ``<split>/data.jsonl`` as the canonical flat training input."""
        lines = []
        for path in self._record_paths(location, split):
            record = self._read_record(path)
            if record is not None:
                lines.append(json.dumps(record, ensure_ascii=False))
        target = location.root / split / "data.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _regenerate_all_records_jsonl(self, location: DatasetLocation) -> None:
        for split in SPLITS:
            self._regenerate_records_jsonl(location, split)

    # ---- validation / normalization -----------------------------------

    def _validate_record(self, location: DatasetLocation, record: dict) -> dict:
        if not isinstance(record, dict):
            raise HTTPException(status_code=422, detail="Record must be a JSON object")
        if location.format == "chat_jsonl":
            return {"messages": self._normalize_chat_messages(record)}
        return self._normalize_instruction_record(record)

    def _normalize_instruction_record(self, record: dict) -> dict:
        # QA-shaped rows map into the instruction shape (question → instruction,
        # context → input, answer → output) rather than getting a third format.
        instruction = self._clean_text(record.get("instruction") or record.get("question"))
        context = self._clean_text(record.get("input") or record.get("context"))
        output = self._clean_text(record.get("output") or record.get("answer"))
        if not instruction:
            raise HTTPException(status_code=422, detail="Record is missing an 'instruction'")
        if not output:
            raise HTTPException(status_code=422, detail="Record is missing a non-empty 'output'")
        normalized = {"instruction": instruction, "output": output}
        if context:
            normalized["input"] = context
        return normalized

    def _normalize_chat_messages(self, record: dict) -> list[dict]:
        raw = record.get("messages")
        if raw is None and isinstance(record.get("conversations"), list):
            raw = self._from_sharegpt(record["conversations"])
        if not isinstance(raw, list) or not raw:
            raise HTTPException(status_code=422, detail="Chat record needs a non-empty 'messages' list")
        messages = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise HTTPException(status_code=422, detail="Each message must be an object")
            role = str(entry.get("role", "")).strip().lower()
            role = _SHAREGPT_ROLE_MAP.get(role, role)
            content = self._clean_text(entry.get("content") or entry.get("value"))
            if role not in CHAT_ROLES:
                raise HTTPException(status_code=422, detail=f"Unsupported message role: {role or '(empty)'}")
            if role == "assistant" and not content:
                raise HTTPException(status_code=422, detail="Assistant messages must have content")
            if not content and role != "assistant":
                # System/user emptiness is a shape error worth surfacing early.
                raise HTTPException(status_code=422, detail=f"{role.capitalize()} message content is empty")
            messages.append({"role": role, "content": content})
        roles = {message["role"] for message in messages}
        if "user" not in roles or "assistant" not in roles:
            raise HTTPException(
                status_code=422,
                detail="Chat records need at least one user and one assistant message",
            )
        return messages

    def _from_sharegpt(self, conversations: list) -> list[dict]:
        messages = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("from", "")).strip().lower()
            role = _SHAREGPT_ROLE_MAP.get(speaker, speaker)
            messages.append({"role": role, "content": turn.get("value", "")})
        return messages

    def _clean_text(self, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    # ---- item projection ----------------------------------------------

    def _record_item_from_path(
        self, location: DatasetLocation, split: str, path: Path, include: bool
    ) -> DatasetItemDetail:
        record = self._read_record(path) or {}
        input_excerpt, output_excerpt, token_estimate, _roles = self._record_excerpts(location, record)
        return DatasetItemDetail(
            id=path.name,
            dataset_id=location.id,
            split=split,  # type: ignore[arg-type]
            filename=path.name,
            media_type="record",
            image_url="",
            text_url=f"/api/datasets/{location.id}/items/{split}/{quote(path.name)}/text",
            text_preview=input_excerpt,
            output_preview=output_excerpt,
            token_estimate=token_estimate,
            width=0,
            height=0,
            annotation_count=0,
            classes=[],
            class_id=None,
            label=None,
            # Records carry their own supervision, so a well-formed record is
            # always "labeled" for the split/EDA counters.
            is_labeled=True,
            annotations=[],
            record=record if include else None,
        )

    def _record_excerpts(
        self, location: DatasetLocation, record: dict
    ) -> tuple[str, str, int, list[str]]:
        if location.format == "chat_jsonl":
            raw_messages = record.get("messages")
            messages = raw_messages if isinstance(raw_messages, list) else []
            roles = [str(message.get("role", "")) for message in messages if isinstance(message, dict)]
            first_user = next(
                (m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"),
                "",
            )
            last_assistant = next(
                (
                    m.get("content", "")
                    for m in reversed(messages)
                    if isinstance(m, dict) and m.get("role") == "assistant"
                ),
                "",
            )
            texts = [str(m.get("content", "")) for m in messages if isinstance(m, dict)]
            return (
                self._excerpt(first_user),
                self._excerpt(last_assistant),
                self._estimate_tokens(*texts),
                roles,
            )
        instruction = str(record.get("instruction", ""))
        context = str(record.get("input", ""))
        output = str(record.get("output", ""))
        return (
            self._excerpt(instruction),
            self._excerpt(output),
            self._estimate_tokens(instruction, context, output),
            [],
        )

    def _excerpt(self, text: str, limit: int = _PREVIEW_LIMIT) -> str:
        compact = " ".join(str(text).split())
        return compact[:limit] + ("..." if len(compact) > limit else "")

    def _estimate_tokens(self, *texts: str) -> int:
        """Whitespace-and-punctuation token estimate; never loads a tokenizer."""
        return sum(len(_TOKEN_PATTERN.findall(text or "")) for text in texts)

    # ---- LLM EDA ------------------------------------------------------

    def _llm_eda_summary(
        self, location: DatasetLocation, dataset_id: str, split: str, splits: list[str]
    ) -> DatasetEdaSummary:
        token_counts: list[int] = []
        char_counts: list[int] = []
        role_counts: dict[str, int] = {}
        empty_output_count = 0
        over_length_count = 0
        seen: set[str] = set()
        duplicate_count = 0
        item_count = 0
        length_threshold = 2048

        for split_name in splits:
            for path in self._record_paths(location, split_name):
                record = self._read_record(path)
                if record is None:
                    continue
                item_count += 1
                input_excerpt, output_excerpt, tokens, roles = self._record_excerpts(location, record)
                token_counts.append(tokens)
                char_counts.append(self._record_char_length(location, record))
                for role in roles:
                    role_counts[role] = role_counts.get(role, 0) + 1
                if not output_excerpt.strip():
                    empty_output_count += 1
                if tokens > length_threshold:
                    over_length_count += 1
                fingerprint = json.dumps(record, sort_keys=True, ensure_ascii=False)
                if fingerprint in seen:
                    duplicate_count += 1
                else:
                    seen.add(fingerprint)

        warnings = []
        if empty_output_count:
            warnings.append(f"{empty_output_count} records have an empty response")
        if over_length_count:
            warnings.append(f"{over_length_count} records exceed ~{length_threshold} estimated tokens")
        if duplicate_count:
            warnings.append(f"{duplicate_count} duplicate records")
        if item_count == 0:
            warnings.append("No records found in this split")

        return DatasetEdaSummary(
            dataset_id=dataset_id,
            split=split,
            split_counts={name: len(self._record_paths(location, name)) for name in SPLITS},
            class_counts={},
            unlabeled_count=empty_output_count,
            missing_annotation_count=0,
            image_count=item_count,
            text_count=item_count,
            item_count=item_count,
            annotation_count=item_count,
            text_length={
                "min_tokens": min(token_counts) if token_counts else None,
                "max_tokens": max(token_counts) if token_counts else None,
                "mean_tokens": round(mean(token_counts), 2) if token_counts else None,
                "min_chars": min(char_counts) if char_counts else None,
                "max_chars": max(char_counts) if char_counts else None,
                "mean_chars": round(mean(char_counts), 2) if char_counts else None,
            },
            role_counts=role_counts,
            duplicate_count=duplicate_count,
            warnings=warnings,
        )

    def _record_char_length(self, location: DatasetLocation, record: dict) -> int:
        if location.format == "chat_jsonl":
            raw_messages = record.get("messages")
            messages = raw_messages if isinstance(raw_messages, list) else []
            return sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
        return sum(
            len(str(record.get(key, ""))) for key in ("instruction", "input", "output")
        )

    # ---- guards -------------------------------------------------------

    def _require_llm(self, location: DatasetLocation) -> None:
        if not self._is_llm_task(location.task_type):
            raise HTTPException(status_code=409, detail="Dataset is not an LLM fine-tuning dataset")

    if TYPE_CHECKING:
        # Provided by sibling mixins on the composed DatasetService facade.
        def _location(self, dataset_id: str) -> DatasetLocation:
            raise NotImplementedError

        def _editable_location(self, dataset_id: str) -> DatasetLocation:
            raise NotImplementedError

        def _is_llm_task(self, task_type: str) -> bool:
            raise NotImplementedError

        def _media_dir(self, task_type: str) -> str:
            raise NotImplementedError

        def _validate_split(self, split: str) -> None:
            raise NotImplementedError

        def _text_upload_rows(self, raw: str, suffix: str) -> list[dict]:
            raise NotImplementedError

        def summary(self, dataset_id: str):
            raise NotImplementedError
