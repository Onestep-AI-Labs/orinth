"""HuggingFace Hub dataset search, preview, and import for LLM fine-tuning.

Search proxies ``huggingface_hub.HfApi.list_datasets`` (a transitive dependency
of transformers). Preview and import read sample rows through the hosted
datasets-server rows API over ``httpx`` — no local ``datasets`` install is
required for the common case, so text imports work on the base extra. Import is
synchronous and row-capped: text rows land in seconds, so no job table is
warranted. All shape detection (alpaca, ShareGPT, messages, QA) is
re-implemented natively; no code is copied from the AGPL-licensed Unsloth Studio.
"""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import HTTPException

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import (
    DatasetHubImportRequest,
    DatasetHubImportResponse,
    DatasetHubPreview,
    DatasetHubSearchResponse,
    DatasetHubSearchResult,
)
from app.services import hub_facets
from app.services.datasets import DatasetService
from app.services.datasets.constants import LLM_FORMATS
from app.services.datasets.prep.detect import detect_record_format
from app.services.datasets.types import DatasetLocation

DATASETS_SERVER = "https://datasets-server.huggingface.co"
# datasets-server caps a single rows request at 100.
_ROWS_PAGE = 100
# Concurrent page fetches during import — enough to keep the request well under
# the proxy timeout without hammering the Hub into a rate limit.
_IMPORT_WORKERS = 6


def _as_text(value: object) -> object:
    """Flatten a Hub cell into the string a record field has to be.

    Columns are not always scalars. `rajpurkar/squad` — the most-downloaded QA
    dataset on the Hub — stores its answer as
    `{"text": ["Denver Broncos"], "answer_start": [177]}`, and handing that dict
    to `_validate_record` raised for every single row: the import "succeeded"
    with 5,000 skipped and nothing imported. The nesting is an artifact of how
    the dataset records answer spans, not something the user asked for.

    So a one-element list unwraps, a struct yields its first string-valued
    member (`text` first, since that is the near-universal name for the payload),
    and anything else is left alone for `_validate_record` to reject honestly.
    `None` passes through untouched — an absent value is a skipped row, which is
    correct, where `"None"` would be a corrupt one.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return None
        return _as_text(value[0])
    if isinstance(value, dict):
        for key in ("text", "value", "content", "answer"):
            if key in value:
                return _as_text(value[key])
        for nested in value.values():
            flattened = _as_text(nested)
            if isinstance(flattened, str):
                return flattened
        return None
    return value


class DatasetHubService:
    def __init__(self, settings: Settings, storage: Storage, dataset_service: DatasetService) -> None:
        self.settings = settings
        self.storage = storage
        self.datasets = dataset_service

    # ---- search --------------------------------------------------------

    def search(
        self,
        query: str | None,
        task: str | None,
        limit: int,
        *,
        modalities: list[str] | None = None,
        formats: list[str] | None = None,
        sizes: list[str] | None = None,
        tasks: list[str] | None = None,
        sort: str = "trending",
    ) -> DatasetHubSearchResponse:
        """Browse the Hub the way huggingface.co does.

        Filtering is pushed to the Hub rather than applied here. The old version
        asked for `limit * 3` results and dropped the non-text ones locally,
        which is wrong in both directions: it wastes most of a request, and a
        filter that only ever sees one page cannot find a match on page four.
        `list_datasets(filter=[...])` matches the same `prefix:value` tags the
        website's own facets use, so what is shown here is what the Hub shows.

        The one thing still filtered locally is `size_categories`, because the
        Hub takes it as a dedicated argument rather than a tag filter — passing
        it through `filter=` silently matches nothing.
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.settings.huggingface_token)
            tag_filters = [f"modality:{value}" for value in (modalities or [])]
            tag_filters += [f"format:{value}" for value in (formats or [])]

            wanted_tasks = list(tasks or [])
            # The legacy single `task` parameter predates the multi-select and is
            # still what the training page links in with.
            if task and task not in wanted_tasks:
                wanted_tasks.append(task)

            infos = api.list_datasets(
                search=query or None,
                filter=tag_filters or None,
                size_categories=sizes or None,
                task_categories=wanted_tasks or None,
                limit=limit,
                # `direction` was removed in huggingface_hub 1.x; every key here
                # already sorts descending, so passing it would only raise.
                sort=hub_facets.SORT_KEYS.get(sort, "trendingScore"),
                full=True,
            )
            return DatasetHubSearchResponse(
                results=[self._result_from_info(info) for info in infos]
            )
        except Exception as exc:  # noqa: BLE001 — Hub errors must not break the local catalog.
            return DatasetHubSearchResponse(results=[], error=self._humanize(exc))

    @staticmethod
    def _result_from_info(info) -> DatasetHubSearchResult:
        tags = list(getattr(info, "tags", []) or [])
        grouped = hub_facets.split_tags(tags)
        modalities = grouped.get("modality", [])
        card = getattr(info, "card_data", None)
        last_modified = getattr(info, "last_modified", None)
        licenses = grouped.get("license", [])
        sizes = grouped.get("size_categories", [])

        return DatasetHubSearchResult(
            hub_id=info.id,
            author=getattr(info, "author", None),
            downloads=int(getattr(info, "downloads", 0) or 0),
            likes=int(getattr(info, "likes", 0) or 0),
            gated=bool(getattr(info, "gated", False)),
            tags=tags[:30],
            updated_at=last_modified.isoformat() if last_modified else None,
            pretty_name=getattr(card, "pretty_name", None) if card else None,
            modalities=modalities,
            formats=grouped.get("format", []),
            task_categories=grouped.get("task_categories", []),
            languages=grouped.get("language", [])[:6],
            license=licenses[0] if licenses else None,
            size_category=sizes[0] if sizes else None,
            # `viewer` is only present on a card that explicitly disables it, so
            # absent means the viewer works — the common case.
            has_viewer=bool(getattr(card, "viewer", True)) if card else True,
            importable=hub_facets.is_importable(modalities),
            trending_score=int(getattr(info, "trending_score", 0) or 0),
            summary=DatasetHubService._summarize(getattr(info, "description", None)),
        )

    @staticmethod
    def _summarize(description: str | None) -> str | None:
        """Flatten a dataset card's opening into one line.

        Cards are markdown and their first hundred characters are usually the
        title repeated inside nested heading whitespace, so collapsing runs of
        whitespace and taking a prefix beats any attempt to parse structure.
        """
        if not description:
            return None
        text = " ".join(str(description).split())
        if not text:
            return None
        return text[:220] + ("…" if len(text) > 220 else "")

    # ---- preview -------------------------------------------------------

    def preview(
        self, hub_id: str, config: str | None, split: str | None, limit: int
    ) -> DatasetHubPreview:
        try:
            configs, splits = self._configs_and_splits(hub_id, config)
            resolved_config = config or (configs[0] if configs else "default")
            resolved_split = split or (splits[0] if splits else "train")
            rows, columns, column_types = self._fetch_rows(
                hub_id, resolved_config, resolved_split, 0, min(limit, _ROWS_PAGE)
            )
            detected_format, mapping = self._detect_format(columns)
            return DatasetHubPreview(
                hub_id=hub_id,
                config=resolved_config,
                split=resolved_split,
                configs=configs,
                splits=splits,
                columns=columns,
                column_types=column_types,
                rows=rows,
                num_rows=self._split_rows(hub_id, resolved_config, resolved_split),
                detected_format=detected_format,
                detected_mapping=mapping,
            )
        except HTTPException as exc:
            # Surface preview failures (gated, unsupported, rate-limited) as a
            # panel message rather than an HTTP error, so the dialog stays usable
            # and import remains possible blind.
            detail = exc.detail if isinstance(exc.detail, str) else "Preview unavailable"
            return DatasetHubPreview(hub_id=hub_id, config=config, split=split, error=detail)
        except Exception as exc:  # noqa: BLE001
            return DatasetHubPreview(hub_id=hub_id, config=config, split=split, error=self._humanize(exc))

    def _configs_and_splits(self, hub_id: str, config: str | None) -> tuple[list[str], list[str]]:
        data = self._get_json(f"{DATASETS_SERVER}/splits", {"dataset": hub_id})
        entries = data.get("splits", []) if isinstance(data, dict) else []
        configs: list[str] = []
        splits: list[str] = []
        for entry in entries:
            cfg = entry.get("config")
            spl = entry.get("split")
            if cfg and cfg not in configs:
                configs.append(cfg)
            if spl and (config is None or cfg == config) and spl not in splits:
                splits.append(spl)
        return configs, splits

    def _fetch_rows(
        self, hub_id: str, config: str, split: str, offset: int, length: int
    ) -> tuple[list[dict], list[str], dict[str, str]]:
        data = self._get_json(
            f"{DATASETS_SERVER}/rows",
            {"dataset": hub_id, "config": config, "split": split, "offset": offset, "length": length},
        )
        features = [feature for feature in data.get("features", []) if "name" in feature]
        columns = [feature["name"] for feature in features]
        column_types = {
            feature["name"]: self._feature_type(feature.get("type")) for feature in features
        }
        rows = [entry.get("row", {}) for entry in data.get("rows", [])]
        return rows, columns, column_types

    @staticmethod
    def _feature_type(spec: object) -> str:
        """Reduce a datasets-server feature spec to one word.

        The spec is a nested structure — `{"dtype": "string"}`, or a `_type` of
        `Sequence`/`ClassLabel`, or a bare list for a list column. The viewer
        shows this beside a column name, so anything longer than a word competes
        with the name itself; the nesting is not information the user needs to
        decide whether a column holds their prompts.
        """
        if isinstance(spec, list):
            return "list"
        if not isinstance(spec, dict):
            return "unknown"
        dtype = spec.get("dtype")
        if isinstance(dtype, str):
            return dtype
        kind = spec.get("_type")
        if kind == "Sequence" or "feature" in spec:
            return "list"
        if isinstance(kind, str):
            return kind.lower()
        # A struct is a bare dict of nested feature specs with no `_type` of its
        # own. Naming it is what tells a user why that column previews as JSON.
        if spec and all(isinstance(nested, dict) for nested in spec.values()):
            return "struct"
        return "unknown"

    def _split_rows(self, hub_id: str, config: str, split: str) -> int | None:
        """Row count for one split, or None when the Hub will not say.

        None and zero are different answers and must not render the same: an
        unsupported dataset reports nothing, and showing that as "0 rows" reads
        as an empty dataset. Failures here are swallowed because a missing count
        must not cost the user their preview.
        """
        try:
            data = self._get_json(f"{DATASETS_SERVER}/size", {"dataset": hub_id})
        except Exception:  # noqa: BLE001 - the count is a nicety, the rows are not
            return None
        sizes = data.get("size", {}) if isinstance(data, dict) else {}
        for entry in sizes.get("splits", []) or []:
            if entry.get("config") == config and entry.get("split") == split:
                value = entry.get("num_rows")
                return int(value) if isinstance(value, int) else None
        return None

    def _get_json(self, url: str, params: dict) -> dict:
        headers = {}
        token = self.settings.huggingface_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = httpx.get(url, params=params, headers=headers, timeout=30.0)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"HuggingFace request failed: {exc}") from exc
        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=response.status_code,
                detail=(
                    f"Access to '{params.get('dataset')}' is gated or restricted. Accept the license "
                    "on huggingface.co and configure a HuggingFace token in platform settings."
                ),
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"HuggingFace datasets-server error ({response.status_code})")
        return response.json()

    # ---- format detection ---------------------------------------------

    def _detect_format(self, columns: list[str]) -> tuple[str | None, dict[str, str]]:
        # Phase 21 moved the body to `prep/detect.py` so hub import and the prep
        # agent cannot drift into disagreeing about what an alpaca file is.
        return detect_record_format(columns)

    # ---- import --------------------------------------------------------

    def import_hub(self, payload: DatasetHubImportRequest) -> DatasetHubImportResponse:
        if payload.task_type != "llm_finetune":
            raise HTTPException(status_code=400, detail="Hub import currently supports llm_finetune datasets only")
        if payload.format not in LLM_FORMATS:
            raise HTTPException(status_code=400, detail="Import format must be instruction_jsonl or chat_jsonl")
        self._validate_mapping(payload)

        configs, splits = self._configs_and_splits(payload.hub_id, payload.config)
        config = payload.config or (configs[0] if configs else "default")
        split = payload.split if payload.split in splits or not splits else (splits[0] if splits else payload.split)
        revision = self._revision(payload.hub_id)

        temp_root = self.storage.datasets / f".import-{uuid4().hex[:10]}"
        dataset_id = self.datasets._new_dataset_id(payload.name or payload.hub_id.split("/")[-1])
        name = self._dedupe_name(payload.project_id, payload.name or payload.hub_id.split("/")[-1])
        try:
            imported, skipped, warnings = self._write_records(temp_root, payload, config, split)
            if imported == 0:
                raise HTTPException(status_code=422, detail="No rows could be imported with this column mapping")
            self.datasets._write_manifest(
                temp_root,
                dataset_id=dataset_id,
                name=name,
                format_name=payload.format,
                project_id=payload.project_id,
                task_type="llm_finetune",
                labels=[],
                metadata={"imported_rows": imported, "skipped_rows": skipped},
                origin="imported_hf",
                origin_ref=f"{payload.hub_id}@{revision}",
            )
            final_root = self.storage.datasets / dataset_id
            # Atomic publish: an interrupted download never leaves a half-dataset
            # in the catalog because the move only happens after every row lands.
            temp_root.rename(final_root)
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise

        return DatasetHubImportResponse(
            dataset=self.datasets.summary(dataset_id),
            imported_rows=imported,
            skipped_rows=skipped,
            warnings=warnings,
        )

    def _validate_mapping(self, payload: DatasetHubImportRequest) -> None:
        mapping = payload.mapping
        errors: list[str] = []
        if payload.format == "chat_jsonl":
            if not (mapping.messages or mapping.conversations):
                errors.append("Map a 'messages' or 'conversations' column for chat data")
        else:
            has_instruction = bool(mapping.instruction and mapping.output)
            has_qa = bool(mapping.question and mapping.answer)
            if not (has_instruction or has_qa):
                errors.append("Map instruction+output or question+answer columns")
        if errors:
            raise HTTPException(status_code=422, detail={"mapping": errors})

    def _write_records(
        self, temp_root: Path, payload: DatasetHubImportRequest, config: str, split: str
    ) -> tuple[int, int, list[str]]:
        self.datasets._create_layout(temp_root, payload.format, "llm_finetune", [])
        location = DatasetLocation(
            id="__import__",
            project_id=payload.project_id,
            name=payload.name or payload.hub_id,
            task_type="llm_finetune",
            format=payload.format,
            source="editable",
            root=temp_root,
            editable=True,
            labels=[],
            metadata={},
        )
        raw_rows = self._fetch_all_rows(payload.hub_id, config, split, payload.max_rows)
        records: list[dict] = []
        skipped = 0
        for row in raw_rows:
            try:
                record = self.datasets._validate_record(location, self._record_from_row(payload, row))
            except Exception:  # noqa: BLE001 — one malformed row must not fail the run.
                skipped += 1
                continue
            records.append(record)

        # Write the per-record files, then the canonical data.jsonl straight from
        # the in-memory list — re-reading thousands of files back off disk was the
        # slow tail that timed the request out.
        for record in records:
            self.datasets._write_record(location, "unassigned", record)
        jsonl = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        (temp_root / "unassigned" / "data.jsonl").write_text(
            jsonl + ("\n" if records else ""), encoding="utf-8"
        )
        warnings = [f"{skipped} rows skipped (unmappable or invalid)"] if skipped else []
        return len(records), skipped, warnings

    def _fetch_all_rows(self, hub_id: str, config: str, split: str, max_rows: int) -> list[dict]:
        """Page the datasets-server rows API concurrently (it caps a page at 100).

        Fetching up to 5000 rows one 100-row page at a time is 50 sequential
        round-trips — slow enough to trip the dev proxy timeout. A small worker
        pool cuts that to a handful of rounds while staying gentle on the Hub.
        """
        offsets = list(range(0, max_rows, _ROWS_PAGE))
        collected: dict[int, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=_IMPORT_WORKERS) as pool:
            futures = {
                pool.submit(
                    self._safe_fetch_rows, hub_id, config, split, offset, min(_ROWS_PAGE, max_rows - offset)
                ): offset
                for offset in offsets
            }
            for future in as_completed(futures):
                collected[futures[future]] = future.result()
        rows: list[dict] = []
        for offset in offsets:
            page = collected.get(offset, [])
            rows.extend(page)
            # A short page means we reached the end of the split; later offsets
            # are past the end and safe to drop.
            if len(page) < _ROWS_PAGE:
                break
        return rows[:max_rows]

    def _safe_fetch_rows(self, hub_id: str, config: str, split: str, offset: int, length: int) -> list[dict]:
        try:
            rows, _columns, _types = self._fetch_rows(hub_id, config, split, offset, length)
            return rows
        except HTTPException:
            # A single failed page must not abort the whole import; the ordered
            # assembly in _fetch_all_rows stops at the first short/empty page.
            return []

    def _record_from_row(self, payload: DatasetHubImportRequest, row: dict) -> dict:
        mapping = payload.mapping
        if payload.format == "chat_jsonl":
            if mapping.messages:
                return {"messages": row.get(mapping.messages)}
            return {"conversations": row.get(mapping.conversations)}
        # Instruction shape, with QA columns mapping into instruction/input/output.
        instruction = row.get(mapping.instruction) if mapping.instruction else row.get(mapping.question)
        output = row.get(mapping.output) if mapping.output else row.get(mapping.answer)
        context = None
        if mapping.input:
            context = row.get(mapping.input)
        elif mapping.context:
            context = row.get(mapping.context)
        record: dict = {"instruction": _as_text(instruction), "output": _as_text(output)}
        if context is not None:
            record["input"] = _as_text(context)
        return record

    def _revision(self, hub_id: str) -> str:
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=self.settings.huggingface_token).dataset_info(hub_id)
            return getattr(info, "sha", None) or "main"
        except Exception:  # noqa: BLE001
            return "main"

    def _dedupe_name(self, project_id: str, name: str) -> str:
        existing = {
            summary.name
            for summary in self.datasets.list_datasets(project_id)
            if summary.project_id == project_id
        }
        if name not in existing:
            return name
        for index in range(2, 100):
            candidate = f"{name} ({index})"
            if candidate not in existing:
                return candidate
        return f"{name} ({uuid4().hex[:4]})"

    def _humanize(self, exc: Exception) -> str:
        message = str(exc).strip()
        return message or exc.__class__.__name__
