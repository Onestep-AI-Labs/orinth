from datetime import datetime
from typing import Any

from app.schemas import JobProgress


def progress_from_artifacts(artifacts: dict[str, Any] | None) -> JobProgress:
    if not artifacts:
        return JobProgress()
    raw = artifacts.get("progress")
    if not raw:
        return JobProgress(logs=artifacts.get("logs", [])[-25:])
    return JobProgress.model_validate(raw)


def make_progress(
    *,
    percent: float,
    processed: int = 0,
    total: int | None = None,
    current_step: str,
    current_item: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    logs: list[str] | None = None,
) -> dict[str, Any]:
    elapsed = 0.0
    eta = None
    now = finished_at or datetime.utcnow()
    if started_at is not None:
        elapsed = max(0.0, (now - started_at).total_seconds())
        if total and processed > 0 and processed < total:
            eta = max(0.0, elapsed * ((total - processed) / processed))

    progress = JobProgress(
        percent=round(max(0.0, min(100.0, percent)), 2),
        processed=processed,
        total=total,
        current_step=current_step,
        current_item=current_item,
        elapsed_seconds=round(elapsed, 2),
        eta_seconds=round(eta, 2) if eta is not None else None,
        logs=(logs or [])[-25:],
        started_at=started_at,
        finished_at=finished_at,
    )
    return progress.model_dump(mode="json")


def append_log(artifacts: dict[str, Any] | None, message: str, limit: int = 100) -> dict[str, Any]:
    updated = dict(artifacts or {})
    logs = list(updated.get("logs", []))
    logs.append(message)
    updated["logs"] = logs[-limit:]
    return updated
