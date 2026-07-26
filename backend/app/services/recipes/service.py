"""Recipe lifecycle: sources, extraction, generation, review, and commit.

Filesystem-only persistence mirroring datasets — recipes are working areas, not
queryable history, so there is no DB table or migration (phase-11 spec). State
lives in ``storage/recipes/<id>/manifest.json`` alongside ``sources/``,
``chunks.json``, and ``records.jsonl``.

Generation runs on a thread executor; the frontend polls recipe status. Because
executor work does not survive a restart, the startup sweep flips any manifest
left in a transient status to ``failed`` (same reason phase-3 fails orphaned
training jobs).
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import (
    OpenRouterModel,
    OpenRouterModelsResponse,
    RecipeCommitRequest,
    RecipeCommitResponse,
    RecipeCreate,
    RecipeGenerateRequest,
    RecipeGenerationSettings,
    RecipeRead,
    RecipeRecord,
    RecipeRecordCreate,
    RecipeRecordPage,
    RecipeRecordUpdate,
    RecipeSourceRead,
)
from app.services.datasets import DatasetService
from app.services.datasets.types import DatasetLocation
from app.services.recipes import generation as gen
from app.services.recipes import openrouter
from app.services.recipes.chunking import chunk_sources
from app.services.recipes.constants import (
    ALLOWED_SUFFIXES,
    MAX_SOURCE_BYTES,
    TRANSIENT_STATUSES,
)
from app.services.recipes.extraction import extract_text


def _now() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


class RecipeService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        dataset_service: DatasetService,
        executor: ThreadPoolExecutor,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.datasets = dataset_service
        self.executor = executor
        # Between-chunk cancellation signals, keyed by recipe id. Set by
        # `cancel`, checked by the generation loop.
        self._cancel: dict[str, threading.Event] = {}

    # ---- lifecycle ----------------------------------------------------

    def create_recipe(self, payload: RecipeCreate) -> RecipeRead:
        recipe_id = self._new_id(payload.name)
        root = self.storage.recipes / recipe_id
        (root / "sources").mkdir(parents=True, exist_ok=True)
        now = _now()
        manifest = {
            "id": recipe_id,
            "project_id": payload.project_id,
            "name": payload.name.strip(),
            "output_format": payload.output_format,
            "status": "draft",
            "sources": [],
            "generation": payload.generation.model_dump(mode="json"),
            "warnings": [],
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        self._write_manifest(root, manifest)
        return self._to_read(manifest)

    def list_recipes(self, project_id: str | None) -> list[RecipeRead]:
        recipes = []
        for manifest in self._all_manifests():
            if project_id and manifest.get("project_id") != project_id:
                continue
            recipes.append(self._to_read(manifest))
        recipes.sort(key=lambda recipe: recipe.created_at, reverse=True)
        return recipes

    def get_recipe(self, recipe_id: str) -> RecipeRead:
        return self._to_read(self._load(recipe_id))

    def delete_recipe(self, recipe_id: str) -> None:
        root = self._root(recipe_id)
        if not root.exists():
            raise HTTPException(status_code=404, detail="Recipe not found")
        self._cancel.pop(recipe_id, None)
        self.storage.delete_owned_path(root)

    # ---- sources ------------------------------------------------------

    async def add_sources(self, recipe_id: str, files: list[UploadFile]) -> RecipeRead:
        manifest = self._load(recipe_id)
        self._require_idle(manifest)
        root = self._root(recipe_id)
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{suffix or file.filename}'. "
                    "Allowed: PDF, DOCX, TXT, MD, CSV, JSONL",
                )
            data = await file.read()
            if len(data) > MAX_SOURCE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{file.filename}' exceeds the 20 MB per-file limit",
                )
            source_id = f"src-{uuid4().hex[:12]}"
            (root / "sources" / f"{source_id}{suffix}").write_bytes(data)
            result = extract_text(file.filename or source_id, data, suffix)
            (root / "sources" / f"{source_id}.txt").write_text(result.text, encoding="utf-8")
            manifest["sources"].append(
                {
                    "id": source_id,
                    "filename": file.filename or f"{source_id}{suffix}",
                    "media_type": suffix.lstrip("."),
                    "characters": result.characters,
                    "pages": result.pages,
                    "excluded": not result.text.strip(),
                    "warnings": result.warnings,
                }
            )
        manifest["status"] = "draft"
        self._touch(manifest)
        self._write_manifest(root, manifest)
        return self._to_read(manifest)

    def delete_source(self, recipe_id: str, source_id: str) -> RecipeRead:
        manifest = self._load(recipe_id)
        self._require_idle(manifest)
        root = self._root(recipe_id)
        before = len(manifest["sources"])
        manifest["sources"] = [s for s in manifest["sources"] if s["id"] != source_id]
        if len(manifest["sources"]) == before:
            raise HTTPException(status_code=404, detail="Source not found")
        for path in (root / "sources").glob(f"{source_id}*"):
            path.unlink(missing_ok=True)
        self._touch(manifest)
        self._write_manifest(root, manifest)
        return self._to_read(manifest)

    # ---- generation ---------------------------------------------------

    def generate(self, recipe_id: str, request: RecipeGenerateRequest) -> RecipeRead:
        manifest = self._load(recipe_id)
        self._require_idle(manifest)
        sources = self._source_texts(recipe_id, manifest)
        if not sources:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found across sources. Upload readable documents first.",
            )

        settings = self._resolve_settings(manifest, request)
        key = self.settings.openrouter_key
        model = settings.model or self.settings.openrouter_model
        if request.mode == "llm" and not key:
            raise HTTPException(
                status_code=400,
                detail="LLM mode needs an OpenRouter API key. Add one in Settings.",
            )
        if request.mode == "llm" and not model:
            raise HTTPException(
                status_code=400,
                detail="LLM mode needs an OpenRouter model. Choose one in Settings or on this recipe.",
            )
        llm_enabled = request.mode == "llm" or (request.mode == "auto" and bool(key and model))

        manifest["generation"] = settings.model_dump(mode="json")
        manifest["status"] = "generating"
        manifest["warnings"] = []
        manifest["error"] = None
        self._touch(manifest)
        self._write_manifest(self._root(recipe_id), manifest)

        cancel = threading.Event()
        self._cancel[recipe_id] = cancel
        output_format = manifest["output_format"]
        self.executor.submit(
            self._run_generation,
            recipe_id,
            sources,
            settings,
            output_format,
            llm_enabled,
            model,
            key,
            cancel,
        )
        return self._to_read(manifest)

    def cancel(self, recipe_id: str) -> RecipeRead:
        manifest = self._load(recipe_id)
        event = self._cancel.get(recipe_id)
        if event is not None:
            event.set()
        return self._to_read(manifest)

    def _run_generation(
        self,
        recipe_id: str,
        sources: list[tuple[str, str]],
        settings: RecipeGenerationSettings,
        output_format: str,
        llm_enabled: bool,
        model: str | None,
        key: str | None,
        cancel: threading.Event,
    ) -> None:
        root = self._root(recipe_id)
        wrapped: list[dict] = []
        warnings: list[str] = []
        error: str | None = None
        try:
            chunks = chunk_sources(sources, settings.chunk_size, settings.chunk_overlap)
            (root / "chunks.json").write_text(
                json.dumps(
                    [{"index": c.index, "source_id": c.source_id, "offset": c.offset} for c in chunks],
                    indent=2,
                ),
                encoding="utf-8",
            )
            llm_active = llm_enabled
            for chunk in chunks:
                if cancel.is_set():
                    break
                produced, llm_active, chunk_warnings = self._records_for_chunk(
                    chunk, settings, output_format, llm_active, model, key
                )
                wrapped.extend(produced)
                warnings.extend(chunk_warnings)
        except Exception as exc:  # noqa: BLE001 — surface any run failure as recipe status.
            error = str(exc) or exc.__class__.__name__
        finally:
            self._write_records(root, wrapped)
            manifest = self._load_raw(root)
            if manifest is not None:
                manifest["status"] = "failed" if error else "ready"
                manifest["warnings"] = warnings
                manifest["error"] = error
                self._touch(manifest)
                self._write_manifest(root, manifest)
            self._cancel.pop(recipe_id, None)

    def _records_for_chunk(
        self,
        chunk,
        settings: RecipeGenerationSettings,
        output_format: str,
        llm_active: bool,
        model: str | None,
        key: str | None,
    ) -> tuple[list[dict], bool, list[str]]:
        count = settings.records_per_chunk
        flavor = settings.prompt_flavor
        if not llm_active or not (model and key):
            raws = gen.rule_records(chunk.text, output_format, flavor, count)
            return self._wrap(chunk, raws, "rules", output_format), llm_active, []

        raw_records, still_active, warnings = self._llm_chunk(
            chunk, settings, output_format, model, key
        )
        if raw_records is None:
            # LLM path gave up for this chunk; scaffold with rules instead.
            raws = gen.rule_records(chunk.text, output_format, flavor, count)
            return self._wrap(chunk, raws, "rules", output_format), still_active, warnings
        return self._wrap(chunk, raw_records[:count], "llm", output_format), still_active, warnings

    def _llm_chunk(
        self, chunk, settings: RecipeGenerationSettings, output_format: str, model: str, key: str
    ) -> tuple[list[dict] | None, bool, list[str]]:
        """Return ``(records | None, llm_still_active, warnings)``.

        ``records is None`` means "fall back to rules for this chunk". A hard API
        failure (rejected key, repeated 429, timeout) flips the remainder of the
        run to rule-based by returning ``llm_still_active=False``.
        """
        try:
            content = openrouter.chat_completion(key, model, *gen.build_prompt(
                chunk.text, output_format, settings.prompt_flavor, settings.records_per_chunk
            ))
        except openrouter.OpenRouterAuthError:
            return None, False, ["OpenRouter key was rejected; remaining chunks used rules"]
        except (openrouter.OpenRouterRateLimitError, openrouter.OpenRouterError) as exc:
            # One retry, then flip the rest of the run to rules.
            try:
                content = openrouter.chat_completion(key, model, *gen.build_prompt(
                    chunk.text, output_format, settings.prompt_flavor, settings.records_per_chunk
                ))
            except openrouter.OpenRouterError:
                return None, False, [f"OpenRouter unavailable ({exc}); remaining chunks used rules"]

        records = gen.parse_llm_records(content, output_format)
        if records is None:
            # One repair retry demanding strict JSON.
            try:
                repaired = openrouter.chat_completion(key, model, *gen.build_repair_prompt(
                    content, output_format, settings.records_per_chunk
                ))
            except openrouter.OpenRouterError:
                return None, False, ["OpenRouter unavailable during repair; remaining chunks used rules"]
            records = gen.parse_llm_records(repaired, output_format)
        if records is None:
            return None, True, [f"Chunk {chunk.index}: LLM output was not valid, used rules"]
        valid = [r for r in (self._normalize(rec, output_format) for rec in records) if r is not None]
        if not valid:
            return None, True, [f"Chunk {chunk.index}: LLM output failed validation, used rules"]
        return valid, True, []

    def _wrap(self, chunk, raws: list[dict], generator: str, output_format: str) -> list[dict]:
        wrapped = []
        for raw in raws:
            record = self._normalize(raw, output_format)
            if record is None:
                continue
            wrapped.append(
                {
                    "record": record,
                    "generator": generator,
                    "source_id": chunk.source_id,
                    "chunk_index": chunk.index,
                }
            )
        return wrapped

    # ---- records review ----------------------------------------------

    def list_records(self, recipe_id: str, page: int, page_size: int) -> RecipeRecordPage:
        self._load(recipe_id)
        rows = self._read_records(self._root(recipe_id))
        start = (page - 1) * page_size
        page_rows = rows[start : start + page_size]
        return RecipeRecordPage(
            records=[self._record_read(index, row) for index, row in enumerate(page_rows, start=start)],
            total=len(rows),
            page=page,
            page_size=page_size,
        )

    def update_record(self, recipe_id: str, index: int, payload: RecipeRecordUpdate) -> RecipeRecord:
        manifest = self._load(recipe_id)
        self._require_idle(manifest)
        root = self._root(recipe_id)
        rows = self._read_records(root)
        if index < 0 or index >= len(rows):
            raise HTTPException(status_code=404, detail="Record not found")
        record = self._normalize(payload.record, manifest["output_format"])
        if record is None:
            raise HTTPException(status_code=422, detail="Record does not match the recipe format")
        rows[index]["record"] = record  # edits keep provenance fields
        self._write_records(root, rows)
        return self._record_read(index, rows[index])

    def add_record(self, recipe_id: str, payload: RecipeRecordCreate) -> RecipeRecord:
        manifest = self._load(recipe_id)
        self._require_idle(manifest)
        root = self._root(recipe_id)
        record = self._normalize(payload.record, manifest["output_format"])
        if record is None:
            raise HTTPException(status_code=422, detail="Record does not match the recipe format")
        rows = self._read_records(root)
        row = {"record": record, "generator": "rules", "source_id": None, "chunk_index": None}
        rows.append(row)
        self._write_records(root, rows)
        return self._record_read(len(rows) - 1, row)

    def delete_records(self, recipe_id: str, indices: list[int]) -> RecipeRead:
        manifest = self._load(recipe_id)
        self._require_idle(manifest)
        root = self._root(recipe_id)
        rows = self._read_records(root)
        drop = {i for i in indices if 0 <= i < len(rows)}
        rows = [row for index, row in enumerate(rows) if index not in drop]
        self._write_records(root, rows)
        self._touch(manifest)
        self._write_manifest(root, manifest)
        return self._to_read(manifest)

    # ---- commit -------------------------------------------------------

    def commit(self, recipe_id: str, payload: RecipeCommitRequest) -> RecipeCommitResponse:
        manifest = self._load(recipe_id)
        self._require_idle(manifest)
        rows = self._read_records(self._root(recipe_id))
        if not rows:
            raise HTTPException(status_code=400, detail="Add at least one record before committing")

        from app.schemas import DatasetCreate, DatasetRecordCreate

        dataset = self.datasets.create_dataset(
            DatasetCreate(
                project_id=manifest["project_id"],
                name=(payload.name or manifest["name"]).strip()[:120] or manifest["name"],
                task_type="llm_finetune",
                format=manifest["output_format"],
                labels=[],
            )
        )
        # Stamp provenance on the manifest the create just wrote.
        location = self.datasets._editable_location(dataset.id)
        self.datasets._update_manifest(location, origin="recipe", origin_ref=recipe_id)

        committed = 0
        for row in rows:
            try:
                # Records land in the `unassigned` inbox so the standard split
                # flow applies (phase-4 convention).
                self.datasets.create_record(
                    dataset.id, DatasetRecordCreate(split="unassigned", record=row["record"])
                )
                committed += 1
            except HTTPException:
                continue
        return RecipeCommitResponse(dataset=self.datasets.summary(dataset.id), committed_records=committed)

    # ---- openrouter ---------------------------------------------------

    def openrouter_models(self) -> OpenRouterModelsResponse:
        key = self.settings.openrouter_key
        if key:
            try:
                models = openrouter.list_models(key)
                return OpenRouterModelsResponse(
                    models=[OpenRouterModel(id=mid, name=name) for mid, name in models],
                    live=True,
                )
            except Exception:  # noqa: BLE001 — never render an empty select.
                pass
        return OpenRouterModelsResponse(
            models=[OpenRouterModel(id=mid, name=name) for mid, name in openrouter.curated_models()],
            live=False,
        )

    # ---- startup sweep ------------------------------------------------

    def reconcile_stale_recipes(self) -> None:
        for manifest in self._all_manifests():
            if manifest.get("status") in TRANSIENT_STATUSES:
                manifest["status"] = "failed"
                manifest["error"] = "Backend restarted before generation completed"
                self._touch(manifest)
                self._write_manifest(self._root(manifest["id"]), manifest)

    # ---- helpers ------------------------------------------------------

    def _resolve_settings(
        self, manifest: dict, request: RecipeGenerateRequest
    ) -> RecipeGenerationSettings:
        current = RecipeGenerationSettings(**manifest.get("generation", {}))
        return current.model_copy(
            update={
                "mode": request.mode,
                "prompt_flavor": request.prompt_flavor or current.prompt_flavor,
                "model": request.model if request.model is not None else current.model,
                "chunk_size": request.chunk_size or current.chunk_size,
                "chunk_overlap": request.chunk_overlap
                if request.chunk_overlap is not None
                else current.chunk_overlap,
                "records_per_chunk": request.records_per_chunk or current.records_per_chunk,
            }
        )

    def _source_texts(self, recipe_id: str, manifest: dict) -> list[tuple[str, str]]:
        root = self._root(recipe_id)
        sources: list[tuple[str, str]] = []
        for source in manifest.get("sources", []):
            if source.get("excluded"):
                continue
            path = root / "sources" / f"{source['id']}.txt"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if text:
                sources.append((source["id"], text))
        return sources

    def _normalize(self, record: dict, output_format: str) -> dict | None:
        """Validate/normalize a candidate against the phase-10 record shape."""
        location = self._record_location(output_format)
        try:
            return self.datasets._validate_record(location, record)
        except HTTPException:
            return None
        except Exception:  # noqa: BLE001 — a malformed candidate must not crash the run.
            return None

    def _record_location(self, output_format: str) -> DatasetLocation:
        return DatasetLocation(
            id="__recipe__",
            project_id="",
            name="recipe",
            task_type="llm_finetune",
            format=output_format,
            source="editable",
            root=Path("."),
            editable=True,
            labels=[],
            metadata={},
        )

    def _record_read(self, index: int, row: dict) -> RecipeRecord:
        return RecipeRecord(
            index=index,
            record=row.get("record", {}),
            generator=row.get("generator", "rules"),
            source_id=row.get("source_id"),
            chunk_index=row.get("chunk_index"),
        )

    def _read_records(self, root: Path) -> list[dict]:
        path = root / "records.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _write_records(self, root: Path, rows: list[dict]) -> None:
        path = root / "records.jsonl"
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        path.write_text(payload + ("\n" if rows else ""), encoding="utf-8")

    def _require_idle(self, manifest: dict) -> None:
        if manifest.get("status") in TRANSIENT_STATUSES:
            raise HTTPException(
                status_code=409,
                detail="Recipe is busy. Wait for the current run to finish or cancel it.",
            )

    def _to_read(self, manifest: dict) -> RecipeRead:
        return RecipeRead(
            id=manifest["id"],
            project_id=manifest.get("project_id", ""),
            name=manifest["name"],
            output_format=manifest["output_format"],
            status=manifest["status"],
            sources=[RecipeSourceRead(**source) for source in manifest.get("sources", [])],
            generation=RecipeGenerationSettings(**manifest.get("generation", {})),
            warnings=manifest.get("warnings", []),
            record_count=self._record_count(manifest["id"]),
            error=manifest.get("error"),
            created_at=manifest["created_at"],
            updated_at=manifest["updated_at"],
        )

    def _record_count(self, recipe_id: str) -> int:
        path = self._root(recipe_id) / "records.jsonl"
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _root(self, recipe_id: str) -> Path:
        return self.storage.recipes / Path(recipe_id).name

    def _load(self, recipe_id: str) -> dict:
        manifest = self._load_raw(self._root(recipe_id))
        if manifest is None:
            raise HTTPException(status_code=404, detail="Recipe not found")
        return manifest

    def _load_raw(self, root: Path) -> dict | None:
        path = root / "manifest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _all_manifests(self) -> list[dict]:
        manifests = []
        if not self.storage.recipes.exists():
            return manifests
        for path in sorted(self.storage.recipes.glob("*/manifest.json")):
            try:
                manifests.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return manifests

    def _write_manifest(self, root: Path, manifest: dict) -> None:
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(root / "manifest.json")

    def _touch(self, manifest: dict) -> None:
        manifest["updated_at"] = _now()

    def _new_id(self, name: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
        slug = "-".join(part for part in slug.split("-") if part)[:40] or "recipe"
        return f"{slug}-{uuid4().hex[:8]}"
