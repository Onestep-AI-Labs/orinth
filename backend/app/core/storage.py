from pathlib import Path
import shutil
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
        self.datasets = self.root / "datasets"
        self.dataset_versions = self.root / "dataset_versions"
        self.previews = self.root / "previews"
        self.model_assets = self.root / "model_assets"
        self.model_downloads = self.model_assets / "downloads"
        self.registry_file = self.root / "model_registry.json"

    def ensure(self) -> None:
        for path in [
            self.root,
            self.uploads,
            self.overlays,
            self.evaluations,
            self.training_runs,
            self.trained_models,
            self.datasets,
            self.dataset_versions,
            self.previews,
            self.model_assets,
            self.model_downloads,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> tuple[str, Path]:
        suffix = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
        upload_id = uuid4().hex
        path = self.uploads / f"{upload_id}{suffix}"
        content = await file.read()
        path.write_bytes(content)
        return upload_id, path

    def overlay_path(self, inference_id: str) -> Path:
        return self.overlays / f"{inference_id}.jpg"

    def media_url(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.root.resolve())
        return f"/media/{relative.as_posix()}"

    def owned_path(self, path: str | Path | None) -> Path | None:
        if not path:
            return None
        resolved = Path(path).resolve()
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
