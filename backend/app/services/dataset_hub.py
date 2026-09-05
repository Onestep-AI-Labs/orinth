"""HuggingFace Hub dataset search, preview, and import.

Search proxies ``huggingface_hub.HfApi.list_datasets`` (a transitive dependency
of transformers). Preview and import read sample rows through the hosted
datasets-server rows API over ``httpx`` — no local ``datasets`` install is
required for the common case, so text imports work on the base extra. All shape
detection (alpaca, ShareGPT, messages, QA) is re-implemented natively; no code is
copied from the AGPL-licensed Unsloth Studio.

There are two ways in, and they answer different questions.

``import_hub`` is the **mapped** import: you say which column is the instruction
and which is the output, and it writes ``llm_finetune`` records. It is
synchronous and row-capped because that is all it has to be — a few thousand
text rows land in seconds.

``ingest_hub`` is the **as-is** import, and it declares nothing. It streams the
split into a draft dataset's ``_staging/`` in the shape the Hub served it — a
JSONL file, or a folder of images named by class — and stops there. That is what
makes image, classification, and summarization datasets importable at all:
before it, ``llm_finetune`` was the only destination, so everything else on the
Hub was tagged "browse only". It runs on the prep executor for the same reason
prep does — downloading twenty thousand images is not a request.

**Downloading is not preparing, and this deliberately does not do the second
one.** It used to hand straight off to the phase-21 agent, which meant a
download that worked perfectly reported itself as a failure whenever the agent
could not name a task — and the agent frequently cannot, because a Hub dataset
is under no obligation to be shaped like something Orinth trains. What the user
asked for is a dataset in their workspace: browsable, readable from a notebook,
and *offered* to the agent by a button rather than subjected to it. So the
import ends at "downloaded", and `POST /datasets/{id}/prep` is one click away on
the dataset itself. Files dropped into the workspace still auto-prepare —
there the user has already said what they have by handing over its structure.
"""

import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
from fastapi import HTTPException

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import (
    DatasetHubImportRequest,
    DatasetHubImportResponse,
    DatasetHubIngestRequest,
    DatasetHubPreview,
    DatasetHubSearchResponse,
    DatasetHubSearchResult,
    DatasetSummary,
)
from app.services import hub_facets
from app.services.datasets import DatasetService
from app.services.datasets.constants import LLM_FORMATS
from app.services.datasets.prep.detect import detect_record_format
from app.services.datasets.prep.staging import staging_root
from app.services.datasets.types import DatasetLocation
from app.services.job_runner import submit_job

if TYPE_CHECKING:  # pragma: no cover
    from app.services.datasets.prep.service import DatasetPrepService

DATASETS_SERVER = "https://datasets-server.huggingface.co"
# datasets-server caps a single rows request at 100.
_ROWS_PAGE = 100
# Concurrent page fetches during import — enough to keep the request well under
# the proxy timeout without hammering the Hub into a rate limit.
_IMPORT_WORKERS = 6
#: Blocks an as-is sample is split into when it does not take the whole split.
#: One request each, so this is the floor on requests *and* on how finely the
#: sample spans a sorted split. Eight covers the shapes that matter (sorted by
#: class, by source file, by date) without turning a small import into a burst.
_SAMPLE_BLOCKS = 8
# Concurrent asset downloads during an as-is ingest. Higher than the page pool
# because each one is a small cached file on a CDN rather than a query.
_ASSET_WORKERS = 8
# A single media cell larger than this is skipped rather than staged: the Hub
# serves cached preview assets, and anything this size is not one.
_MAX_ASSET_BYTES = 32 * 1024 * 1024
#: datasets-server feature `_type` values that carry a downloadable file.
_MEDIA_TYPES = {"image", "audio", "video"}


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
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        dataset_service: DatasetService,
        prep_service: "DatasetPrepService | None" = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.datasets = dataset_service
        #: Set by the container after both services exist. The as-is ingest is
        #: the prep agent with a different source of files, so it borrows the
        #: agent's executor and its manifest progress fields rather than growing
        #: a second job kind that means the same thing.
        self.prep = prep_service

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

    def _fetch_page(
        self, hub_id: str, config: str, split: str, offset: int, length: int
    ) -> tuple[list[dict], list[dict]]:
        """One page of rows plus the raw feature specs that describe them.

        The specs are kept rather than reduced to a type word here because the
        as-is ingest needs what `_feature_type` throws away: a `ClassLabel`'s
        `names` (the row carries an integer index, and a folder called `3` is not
        a class) and which columns are `Image`.
        """
        data = self._get_json(
            f"{DATASETS_SERVER}/rows",
            {"dataset": hub_id, "config": config, "split": split, "offset": offset, "length": length},
        )
        features = [feature for feature in data.get("features", []) if "name" in feature]
        rows = [entry.get("row", {}) for entry in data.get("rows", [])]
        return rows, features

    def _fetch_rows(
        self, hub_id: str, config: str, split: str, offset: int, length: int
    ) -> tuple[list[dict], list[str], dict[str, str]]:
        rows, features = self._fetch_page(hub_id, config, split, offset, length)
        columns = [feature["name"] for feature in features]
        column_types = {
            feature["name"]: self._feature_type(feature.get("type")) for feature in features
        }
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
        headers = self._headers()
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

    # ---- as-is ingest --------------------------------------------------

    def ingest_hub(
        self, payload: DatasetHubIngestRequest, *, allowed_task_types: list[str] | None = None
    ) -> DatasetSummary:
        """Download a split as it is into a draft dataset. Nothing else.

        Returns as soon as the draft exists; the download runs on the prep
        executor and reports through `/prep/status`, which the studio already
        polls. When it finishes the dataset is a draft holding its files — the
        Overview tab then offers "Prepare with Orinth", and the notebook can read
        the files either way.

        `allowed_task_types` is accepted and unused: the caller has it, the
        preparation that would consume it is now a separate request, and taking
        it here keeps the route from having to know that.
        """
        del allowed_task_types
        if self.prep is None:  # pragma: no cover - wired in `container.py`
            raise HTTPException(status_code=503, detail="The prep agent is not available.")

        configs, splits = self._configs_and_splits(payload.hub_id, payload.config)
        config = payload.config or (configs[0] if configs else "default")
        split = payload.split if payload.split in splits else (splits[0] if splits else "train")

        name = self._dedupe_name(
            payload.project_id, payload.name or payload.hub_id.split("/")[-1]
        )
        dataset_id = self.prep.create_draft(project_id=payload.project_id, name=name)
        # Provenance now, not after the run: the catalog groups by `origin`, and
        # a dataset that spends two minutes downloading should already be under
        # "Imported from HuggingFace" while it does.
        self._mark_imported(dataset_id, payload.hub_id)
        self.prep._step(
            dataset_id,
            "planning",
            "staging",
            f"Fetching {payload.hub_id} from HuggingFace",
            0.0,
        )

        args = (dataset_id, payload, config, split)
        if self.prep.executor is None:
            # Desktop and test path: `DatasetPrepService` runs inline there too,
            # so the whole ingest is synchronous and the caller gets a finished
            # dataset back.
            self._ingest_tracked(*args)
        else:
            submit_job(self.prep.executor, self._ingest_tracked, *args)
        return self.datasets.summary(dataset_id)

    def _mark_imported(self, dataset_id: str, hub_id: str) -> None:
        location = self.datasets._location(dataset_id)
        self.datasets._update_manifest(
            location,
            origin="imported_hf",
            origin_ref=f"{hub_id}@{self._revision(hub_id)}",
        )

    def _ingest_tracked(
        self,
        dataset_id: str,
        payload: DatasetHubIngestRequest,
        config: str,
        split: str,
    ) -> None:
        """Download the split. Nothing awaits this, so it records its own end."""
        assert self.prep is not None
        try:
            staged, noun = self._stage_split(dataset_id, payload, config, split)
        except Exception as error:  # noqa: BLE001 - a background run must say why it stopped
            self.prep._record_failure(dataset_id, f"The download failed: {error}")
            return
        if staged == 0:
            self.prep._record_failure(
                dataset_id,
                f"HuggingFace returned no rows for {payload.hub_id} ({config}/{split}).",
            )
            return
        # `draft`, not `ready`: the files are here and nothing has been decided
        # about them. That is a complete, correct outcome — readiness renders it
        # as "not prepared yet" with the agent one button away, rather than as a
        # run that failed.
        self.prep._step(
            dataset_id,
            "draft",
            "idle",
            f"Downloaded {staged:,} {noun} from {payload.hub_id}",
            1.0,
        )

    def _stage_split(
        self,
        dataset_id: str,
        payload: DatasetHubIngestRequest,
        config: str,
        split: str,
    ) -> tuple[int, str]:
        """Write the split into `_staging/` in whatever shape it arrived.

        Returns `(count, noun)` — "rows" or "files" — so the status sentence can
        name what actually landed rather than guessing from the dataset.

        Two shapes, decided by the features rather than by the dataset's tags: a
        split with a media column becomes a folder tree named by class, which is
        what the agent reads as image classification; everything else becomes one
        JSONL file, which is what it reads as rows. In both cases `ClassLabel`
        integers are resolved to their names first — a folder called `3` is not a
        class, and neither is a label column full of `3`.
        """
        assert self.prep is not None
        root = self.datasets._location(dataset_id).root
        staging = staging_root(root)
        staging.mkdir(parents=True, exist_ok=True)

        total = self._split_rows(payload.hub_id, config, split)
        wanted = min(payload.max_rows, total) if total else payload.max_rows
        # The download owns the bar's first third; `run(progress_floor=…)` below
        # picks it up from there, so the readout only ever moves forward.
        # The download owns the whole bar: nothing runs after it now that
        # preparing is a separate request.
        tick = self.prep.ticker(dataset_id, "planning", "staging", floor=0.0, ceiling=1.0)

        first, features = self._fetch_page(payload.hub_id, config, split, 0, _ROWS_PAGE)
        if not first:
            return 0, "rows"
        media_column = next(
            (
                feature["name"]
                for feature in features
                if _feature_kind(feature.get("type")) in _MEDIA_TYPES
            ),
            None,
        )
        labels = {
            feature["name"]: _class_names(feature.get("type"))
            for feature in features
            if _feature_kind(feature.get("type")) == "classlabel"
        }

        if media_column:
            return (
                self._stage_assets(
                    payload, config, split, first, media_column, labels, wanted, total, staging, tick
                ),
                "files",
            )
        return (
            self._stage_jsonl(payload, config, split, first, labels, wanted, total, staging, tick),
            "rows",
        )

    def _decode(self, row: dict, labels: dict[str, list[str]]) -> dict:
        """Replace every `ClassLabel` index in a row with its name."""
        if not labels:
            return row
        decoded = dict(row)
        for column, names in labels.items():
            value = decoded.get(column)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 0 <= value < len(names):
                decoded[column] = names[value]
        return decoded

    def _pages(
        self,
        payload: DatasetHubIngestRequest,
        config: str,
        split: str,
        first: list[dict],
        wanted: int,
        total: int | None = None,
    ):
        """Yield `wanted` rows, spread across the split rather than off its head.

        Taking a prefix is the obvious implementation and it is wrong for a large
        class of real datasets. `stanfordnlp/imdb` stores its train split sorted
        by label — rows 0-12,499 are `neg` and 12,500-24,999 are `pos` — so any
        head sample of a perfectly balanced dataset contains exactly one class,
        and detection then refuses to call a single-valued column a taxonomy.
        The import looked like it had failed on a dataset Orinth trains happily.
        Sorting by class, by source file, or by date all have this property, so
        the sample walks the whole split.

        The sample is taken as a handful of *blocks* spread across the split,
        not row by row. The rows API caps a page at 100, so picking individual
        rows means one request per two or three of them — 250 requests for 600
        rows, which datasets-server answers with a 502. Blocks cost one request
        each: `_SAMPLE_BLOCKS` of them at minimum, more only when the request is
        large enough to need more pages anyway. Reading off the head costs the
        same number of requests and misses half the dataset.

        A full take (`wanted >= total`) pages straight through, because then
        there is nothing to choose between. A block that fails is skipped rather
        than fatal — one bad response out of eight should cost an eighth of the
        sample, not the import.
        """
        if not total or wanted >= total:
            yield from self._sequential_pages(payload, config, split, first, wanted)
            return

        # Enough blocks to span the split, and enough pages to hold the request.
        blocks = min(max(_SAMPLE_BLOCKS, -(-wanted // _ROWS_PAGE)), total)
        per_block = min(_ROWS_PAGE, max(1, -(-wanted // blocks)))
        stride = total / blocks
        served = 0
        for index in range(blocks):
            offset = min(int(index * stride), max(0, total - per_block))
            if offset + per_block <= len(first):
                rows = first[offset : offset + per_block]
            else:
                rows = self._safe_fetch_page(payload.hub_id, config, split, offset, per_block)
            for row in rows:
                if served >= wanted:
                    return
                yield row
                served += 1

    def _safe_fetch_page(
        self, hub_id: str, config: str, split: str, offset: int, length: int = _ROWS_PAGE
    ) -> list[dict]:
        """One page, or an empty one. A transient upstream error is not the import."""
        try:
            rows, _features = self._fetch_page(hub_id, config, split, offset, length)
            return rows
        except HTTPException:
            return []


    def _sequential_pages(
        self, payload: DatasetHubIngestRequest, config: str, split: str, first: list[dict], wanted: int
    ):
        """Rows from `first`, then page the rest until the split runs out."""
        yield from first[:wanted]
        seen = min(len(first), wanted)
        offset = len(first)
        while seen < wanted:
            rows, _features = self._fetch_page(
                payload.hub_id, config, split, offset, min(_ROWS_PAGE, wanted - seen)
            )
            if not rows:
                return
            for row in rows:
                if seen >= wanted:
                    return
                yield row
                seen += 1
            offset += len(rows)
            if len(rows) < _ROWS_PAGE:
                return

    def _stage_jsonl(
        self,
        payload: DatasetHubIngestRequest,
        config: str,
        split: str,
        first: list[dict],
        labels: dict[str, list[str]],
        wanted: int,
        total: int | None,
        staging: Path,
        tick,
    ) -> int:
        """One JSONL file, streamed. Every column the Hub served, untouched."""
        target = staging / f"{_safe_segment(payload.hub_id.split('/')[-1], 'dataset')}-{split}.jsonl"
        written = 0
        with target.open("w", encoding="utf-8") as handle:
            for row in self._pages(payload, config, split, first, wanted, total):
                handle.write(json.dumps(self._decode(row, labels), ensure_ascii=False) + "\n")
                written += 1
                tick(written, wanted, f"Downloaded {written:,} of {wanted:,} rows")
        tick(written, written, f"Downloaded {written:,} rows")
        return written

    def _stage_assets(
        self,
        payload: DatasetHubIngestRequest,
        config: str,
        split: str,
        first: list[dict],
        media_column: str,
        labels: dict[str, list[str]],
        wanted: int,
        total: int | None,
        staging: Path,
        tick,
    ) -> int:
        """A folder per class, an image per row — the layout detection reads.

        Nothing else is written beside them. An earlier version also dropped the
        row's other columns into a `metadata.jsonl` sidecar "in case", and that
        one file made `_detect_record_files` — which runs before the image rule —
        classify a folder of 150 pictures as a table of rows. A file nothing
        reads is not free; it is a signal, and detection reads signals.
        """
        label_column = next(iter(labels), None)
        written = 0
        seen = 0

        with ThreadPoolExecutor(max_workers=_ASSET_WORKERS) as pool:
            pending: dict = {}
            for index, row in enumerate(
                self._pages(payload, config, split, first, wanted, total)
            ):
                decoded = self._decode(row, labels)
                url = _asset_url(row.get(media_column))
                if url is None:
                    continue
                folder = (
                    _safe_segment(str(decoded.get(label_column, "")), "unlabeled")
                    if label_column
                    else "unlabeled"
                )
                destination = staging / folder / f"{index:07d}{self._suffix_for(url)}"
                pending[pool.submit(self._download_asset, url, destination)] = index

            for future in as_completed(pending):
                seen += 1
                if future.result():
                    written += 1
                tick(seen, len(pending), f"Downloaded {seen:,} of {len(pending):,} files")

        tick(written, written, f"Downloaded {written:,} files")
        return written

    @staticmethod
    def _suffix_for(url: str) -> str:
        suffix = Path(url.split("?")[0]).suffix.lower()
        return suffix if 1 < len(suffix) <= 5 else ".jpg"

    def _download_asset(self, url: str, destination: Path) -> bool:
        """Fetch one cached asset. A failure is skipped, never fatal to the run."""
        try:
            response = httpx.get(url, headers=self._headers(), timeout=30.0, follow_redirects=True)
            if response.status_code >= 400 or len(response.content) > _MAX_ASSET_BYTES:
                return False
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                shutil.copyfileobj(io.BytesIO(response.content), handle)
            return True
        except (httpx.HTTPError, OSError):
            return False

    def _headers(self) -> dict[str, str]:
        token = self.settings.huggingface_token
        return {"Authorization": f"Bearer {token}"} if token else {}

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


# --- as-is ingest helpers -----------------------------------------------------
#
# Module-level because they are pure functions of a datasets-server payload and
# have nothing to do with the service's settings, storage, or dataset access.


def _feature_kind(spec: object) -> str:
    """The feature's `_type`, lowercased — `image`, `classlabel`, `value`, …"""
    if not isinstance(spec, dict):
        return ""
    kind = spec.get("_type")
    return kind.lower() if isinstance(kind, str) else ""


def _class_names(spec: object) -> list[str]:
    """A `ClassLabel`'s label vocabulary, in index order."""
    if not isinstance(spec, dict):
        return []
    names = spec.get("names")
    if isinstance(names, list) and all(isinstance(entry, str) for entry in names):
        return names
    return []


def _asset_url(cell: object) -> str | None:
    """The `src` of a media cell, if this cell is one.

    datasets-server serves image/audio columns as `{"src": "https://…", …}`
    pointing at a cached asset. A list column of them yields its first entry;
    anything else is not media.
    """
    if isinstance(cell, dict):
        src = cell.get("src")
        return src if isinstance(src, str) and src.startswith("http") else None
    if isinstance(cell, list) and cell:
        return _asset_url(cell[0])
    return None


def _safe_segment(value: str, fallback: str) -> str:
    """A folder or file name that cannot escape staging or surprise a filesystem."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in str(value).strip()
    ).strip("-.")
    return cleaned[:64] or fallback
