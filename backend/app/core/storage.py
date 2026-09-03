import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings


class Storage:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.storage_path
        self.uploads = self.root / "uploads"
        self.overlays = self.root / "overlays"
        self.evaluations = self.root / "evaluations"
        self.training_runs = self.root / "training_runs"
        self.trained_models = self.root / "trained_models"
        self.uploaded_models = self.root / "uploaded_models"
        self.datasets = self.root / "datasets"
        self.dataset_versions = self.root / "dataset_versions"
        self.recipes = self.root / "recipes"
        self.previews = self.root / "previews"
        # Phase 17: one directory per architecture holding a JSON snapshot of
        # every save, so a graph can be recovered after a bad edit.
        self.architectures = self.root / "architectures"
        self.model_assets = self.root / "model_assets"
        self.model_downloads = self.model_assets / "downloads"
        self.registry_file = self.root / "model_registry.json"
        # Phase 15: converter tooling cache and the llama.cpp serving state file
        # (pid/port of the managed subprocess, for orphan reaping).
        self.tools = self.root / "tools"
        self.serving = self.root / "serving"
        self.serving_state_file = self.serving / "state.json"
        # Persisted serving preferences (last models directory, etc.).
        self.serving_config_file = self.serving / "config.json"
        # GGUF/MLX models downloaded from the Hub for serving (phase 15).
        self.serving_models = self.serving / "models"
        # Phase 22: one directory per notebook holding its manifest, its
        # `.ipynb`, and its run artifacts. `jupyter-server` is rooted here, so a
        # kernel's filesystem view is exactly this tree and nothing above it.
        self.notebooks = self.root / "notebooks"
        # Kernelspecs provisioned for the managed server. Kept beside the
        # notebooks rather than in the user's `~/.jupyter`, so the workspace
        # stays self-contained and uninstalling is `rm -rf storage/`.
        self.notebooks_jupyter = self.notebooks / ".jupyter"

    def ensure(self) -> None:
        for path in [
            self.root,
            self.uploads,
            self.overlays,
            self.evaluations,
            self.training_runs,
            self.trained_models,
            self.uploaded_models,
            self.datasets,
            self.dataset_versions,
            self.recipes,
            self.previews,
            self.architectures,
            self.model_assets,
            self.model_downloads,
            self.tools,
            self.serving,
            self.serving_models,
            self.notebooks,
            self.notebooks_jupyter,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> tuple[str, Path]:
        suffix = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
        upload_id = uuid4().hex
        path = self.uploads / f"{upload_id}{suffix}"
        content = await file.read()
        path.write_bytes(content)
        return upload_id, path

    def save_text_input(self, text: str, suffix: str = ".txt") -> tuple[str, Path]:
        upload_id = uuid4().hex
        path = self.uploads / f"{upload_id}{suffix}"
        path.write_text(text, encoding="utf-8")
        return upload_id, path

    def overlay_path(self, inference_id: str) -> Path:
        return self.overlays / f"{inference_id}.jpg"

    def media_url(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.root.resolve())
        return f"/media/{relative.as_posix()}"

    def owned_path(self, path: str | Path | None) -> Path | None:
        if not path:
            return None
        raw_path = Path(path)
        resolved = raw_path.resolve() if raw_path.is_absolute() else (self.root / raw_path).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError:
            return None
        return resolved

    def delete_owned_path(self, path: str | Path | None) -> bool:
        owned = self.owned_path(path)
        if owned is None or not owned.exists():
            return False
        if owned.is_dir():
            shutil.rmtree(owned)
        else:
            owned.unlink()
        return True
