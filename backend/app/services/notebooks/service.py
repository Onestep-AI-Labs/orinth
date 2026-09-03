"""Notebooks on disk: one directory each, a manifest, and an `.ipynb`.

Filesystem-native for the same reason datasets and recipes are — a notebook is a
document, and a document that is also a database row has two truths that drift.
There is no table and no migration here, and the reader looking for the Alembic
line in the spec's Validation section will not find one: a kernel does not
survive a restart, so there is nothing for `reconcile_stale_jobs` to reconcile,
and a row would answer nothing the manifest does not.

**This module never imports `nbformat`.** `cell_count` comes from reading the
`.ipynb` as plain JSON, because the file *is* JSON and pulling in the Jupyter
stack to count a list would put it in the API process — the one place the whole
design keeps it out of.
"""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.core.config import Settings
from app.core.defaults import DEFAULT_PROJECT_ID
from app.core.storage import Storage
from app.schemas import (
    NotebookCreate,
    NotebookSummary,
    NotebookTemplate,
    NotebookUpdate,
)

MANIFEST = "manifest.json"
NOTEBOOK_FILE = "notebook.ipynb"
RUNS_DIRNAME = "runs"

#: Templates ship in the repo rather than in `storage/`: they are source, edited
#: by us and tracked in git, where `storage/` is generated and ignored.
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    return _SLUG.sub("-", value.strip().lower()).strip("-")[:48] or "notebook"


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class NotebookService:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage

    # ---- paths ---------------------------------------------------------

    @property
    def root(self) -> Path:
        return self.storage.notebooks

    def directory(self, notebook_id: str) -> Path:
        """The notebook's directory, refusing anything that escapes the root.

        `notebook_id` arrives from a URL path segment, which makes it
        attacker-controlled in the same way an upload filename is. A `..` here
        would read or delete outside the workspace.
        """
        candidate = (self.root / notebook_id).resolve()
        if candidate.parent != self.root.resolve():
            raise HTTPException(status_code=404, detail=f"No notebook '{notebook_id}'")
        return candidate

    def notebook_file(self, notebook_id: str) -> Path:
        return self.directory(notebook_id) / NOTEBOOK_FILE

    def runs_dir(self, notebook_id: str) -> Path:
        return self.directory(notebook_id) / RUNS_DIRNAME

    def relative_path(self, notebook_id: str) -> str:
        """Path as `jupyter-server` sees it, rooted at `storage/notebooks/`."""
        return f"{notebook_id}/{NOTEBOOK_FILE}"

    # ---- read ----------------------------------------------------------

    def list_notebooks(self, project_id: str | None = None) -> list[NotebookSummary]:
        found: list[NotebookSummary] = []
        if not self.root.is_dir():
            return found
        for directory in sorted(self.root.iterdir()):
            # `.jupyter` holds kernelspecs, not a notebook.
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            summary = self._summary_if_present(directory.name)
            if summary is None:
                continue
            if project_id and summary.project_id != project_id:
                continue
            found.append(summary)
        # Newest-modified first: the notebook you were last in is the one you
        # almost always want next.
        # A notebook whose manifest predates `updated_at` sorts last. The
        # sentinel is a fixed epoch rather than `datetime.min`, which ruff's
        # DTZ901 rejects and which compares badly against the naive timestamps
        # `_now()` writes.
        return sorted(
            found,
            key=lambda item: item.updated_at or datetime(1970, 1, 1),  # noqa: DTZ001
            reverse=True,
        )

    def _summary_if_present(self, notebook_id: str) -> NotebookSummary | None:
        manifest_path = self.root / notebook_id / MANIFEST
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        cells, valid = self._cell_count(self.root / notebook_id / NOTEBOOK_FILE)
        return NotebookSummary(
            id=notebook_id,
            project_id=manifest.get("project_id") or DEFAULT_PROJECT_ID,
            name=manifest.get("name") or notebook_id,
            path=self.relative_path(notebook_id),
            tags=list(manifest.get("tags") or []),
            cell_count=cells,
            created_at=manifest.get("created_at"),
            updated_at=manifest.get("updated_at"),
            valid=valid,
        )

    def summary(self, notebook_id: str) -> NotebookSummary:
        summary = self._summary_if_present(self.directory(notebook_id).name)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No notebook '{notebook_id}'")
        return summary

    @staticmethod
    def _cell_count(path: Path) -> tuple[int, bool]:
        """Cells, and whether the file parsed.

        A corrupt `.ipynb` returns `(0, False)` rather than raising: the notebook
        still has to list, because hiding it would make a broken file look like a
        deleted one and the user cannot fix what they cannot see.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0, False
        cells = payload.get("cells")
        if not isinstance(cells, list):
            return 0, False
        return len(cells), True

    # ---- write ---------------------------------------------------------

    def create(self, payload: NotebookCreate) -> NotebookSummary:
        notebook_id = self._new_id(payload.name)
        directory = self.root / notebook_id
        (directory / RUNS_DIRNAME).mkdir(parents=True, exist_ok=True)

        content = self._template_content(payload.template_id)
        (directory / NOTEBOOK_FILE).write_text(
            json.dumps(content, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        now = _now()
        self._write_manifest(
            directory,
            {
                "id": notebook_id,
                "name": payload.name.strip(),
                "project_id": payload.project_id,
                "tags": [],
                "template_id": payload.template_id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            },
        )
        return self.summary(notebook_id)

    def update(self, notebook_id: str, payload: NotebookUpdate) -> NotebookSummary:
        directory = self.directory(notebook_id)
        manifest = self._read_manifest(directory)
        if payload.name is not None:
            manifest["name"] = payload.name.strip()
        if payload.tags is not None:
            manifest["tags"] = list(payload.tags)
        manifest["updated_at"] = _now().isoformat()
        self._write_manifest(directory, manifest)
        return self.summary(notebook_id)

    def duplicate(self, notebook_id: str) -> NotebookSummary:
        """Copy the document, not the history.

        Runs stay behind deliberately: they record what happened in *that*
        notebook, and carrying them into a copy would attribute someone else's
        numbers to a run that never executed.
        """
        source = self.directory(notebook_id)
        manifest = self._read_manifest(source)
        name = f"{manifest.get('name') or notebook_id} copy"
        new_id = self._new_id(name)
        target = self.root / new_id
        (target / RUNS_DIRNAME).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / NOTEBOOK_FILE, target / NOTEBOOK_FILE)
        now = _now()
        self._write_manifest(
            target,
            {
                **manifest,
                "id": new_id,
                "name": name,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            },
        )
        return self.summary(new_id)

    def delete(self, notebook_id: str) -> None:
        shutil.rmtree(self.directory(notebook_id), ignore_errors=True)

    def touch(self, notebook_id: str) -> None:
        """Bump `updated_at` after the kernel saved through the proxy.

        `jupyter-server` writes the `.ipynb` directly, so without this the
        manifest's timestamp would only ever reflect a rename and the list would
        sort by the wrong thing.
        """
        try:
            directory = self.directory(notebook_id)
            manifest = self._read_manifest(directory)
        except HTTPException:
            return
        manifest["updated_at"] = _now().isoformat()
        self._write_manifest(directory, manifest)

    # ---- templates -----------------------------------------------------

    def templates(self) -> list[NotebookTemplate]:
        found: list[NotebookTemplate] = []
        if not TEMPLATE_DIR.is_dir():
            return found
        for path in sorted(TEMPLATE_DIR.glob("*.ipynb")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            meta = (payload.get("metadata") or {}).get("orinth") or {}
            found.append(
                NotebookTemplate(
                    id=path.stem,
                    name=meta.get("name") or path.stem.replace("-", " ").title(),
                    description=meta.get("description") or "",
                    task_types=list(meta.get("task_types") or []),
                )
            )
        return found

    def _template_content(self, template_id: str | None) -> dict:
        if template_id:
            path = TEMPLATE_DIR / f"{Path(template_id).name}.ipynb"
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"No template '{template_id}'")
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                raise HTTPException(
                    status_code=500, detail=f"Template '{template_id}' is unreadable"
                ) from error
        return blank_notebook()

    # ---- manifest ------------------------------------------------------

    def _new_id(self, name: str) -> str:
        return f"{_slug(name)}-{uuid4().hex[:8]}"

    @staticmethod
    def _read_manifest(directory: Path) -> dict:
        path = directory / MANIFEST
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No notebook '{directory.name}'")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise HTTPException(
                status_code=500, detail=f"Notebook '{directory.name}' has an unreadable manifest"
            ) from error

    @staticmethod
    def _write_manifest(directory: Path, manifest: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MANIFEST).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def blank_notebook() -> dict:
    """An empty nbformat 4.5 document.

    Written by hand rather than with `nbformat.v4.new_notebook()` for the reason
    in the module docstring: this runs in the API process. The schema is stable
    and public, and the file is four keys.
    """
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": uuid4().hex[:8],
                "metadata": {},
                "outputs": [],
                "source": ["import orinth\n", "\n", "orinth.datasets.list()"],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Orinth", "language": "python", "name": "orinth"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
