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
        self.registry_file = self.root / "model_registry.json"

    def ensure(self) -> None:
        for path in [
            self.root,
            self.uploads,
            self.overlays,
            self.evaluations,
            self.training_runs,
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
