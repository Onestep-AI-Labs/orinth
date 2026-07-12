"""Shared job-execution lifecycle and list/delete helpers.

Training, evaluation, and inference jobs each open their own `SessionLocal`,
flip `status` to `running`, run their domain-specific work, and persist a
`status="failed"` + `error` message on any exception, then close the
session. Their `list_jobs`/`delete_jobs` queries duplicate the same
`or_(project_id == x, project_id.is_(None))` filter and the
`{deleted, blocked, missing}` return shape. This module extracts both
skeletons so each service only supplies what actually differs between them:
its start-of-job log/progress format, its domain execution logic, and (for
delete) its per-row deletability check and storage cleanup.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.services.job_progress import append_log, make_progress


def utcnow() -> datetime:
    """Return a naive UTC timestamp, matching the job models' existing convention."""

    return datetime.now(UTC).replace(tzinfo=None)


def run_job_lifecycle(
    session_factory: Callable[[], Session],
    model: Any,
    job_id: str,
    *,
    on_start: Callable[[Session, Any, datetime], None],
    execute: Callable[[Session, Any, datetime], None],
    on_cleanup: Callable[[str], None] | None = None,
) -> None:
    """Run a background job with the shared session/status/failure skeleton.

    Opens a session via ``session_factory``, fetches ``job_id`` (returning
    early if the row is gone), transitions it to ``running``, calls
    ``on_start`` to record the service-specific start log/progress, commits,
    then hands off to ``execute`` for the domain-specific work (which is
    responsible for its own commits and its own terminal status transition).
    Any exception raised by ``on_start`` or ``execute`` is caught and
    persisted as ``status="failed"`` with the exception message, exactly as
    every service previously did inline. ``on_cleanup`` runs in the
    ``finally`` block, before the session closes, so per-service teardown
    (e.g. training clearing its active-process registry) still happens even
    when the job row has disappeared.
    """

    db = session_factory()
    started_at = utcnow()
    try:
        job = db.get(model, job_id)
        if job is None:
            return

        job.status = "running"
        on_start(db, job, started_at)
        job.updated_at = utcnow()
        db.commit()

        execute(db, job, started_at)
    except Exception as exc:  # noqa: BLE001 - background jobs must persist errors
        job = db.get(model, job_id)
        if job is not None:
            artifacts = append_log(job.artifacts, f"Failed: {exc}")
            artifacts["progress"] = make_progress(
                percent=100,
                current_step="Failed",
                started_at=started_at,
                finished_at=utcnow(),
                logs=artifacts["logs"],
            )
            job.status = "failed"
            job.error = str(exc)
            job.artifacts = artifacts
            job.updated_at = utcnow()
            db.commit()
    finally:
        if on_cleanup is not None:
            on_cleanup(job_id)
        db.close()


def list_jobs(
    db: Session,
    model: Any,
    *,
    limit: int = 25,
    project_id: str | None = None,
) -> list[Any]:
    """Shared list query: newest first, optionally scoped to a project.

    Matches the previously duplicated per-service filter: rows for the given
    ``project_id`` plus any rows with no project (``project_id IS NULL``)
    are included so legacy/global rows keep showing up.
    """

    query = select(model).order_by(model.created_at.desc()).limit(limit)
    if project_id:
        query = (
            select(model)
            .where(or_(model.project_id == project_id, model.project_id.is_(None)))
            .order_by(model.created_at.desc())
            .limit(limit)
        )
    return list(db.scalars(query).all())


def delete_jobs(
    db: Session,
    model: Any,
    *,
    ids: list[str] | None = None,
    project_id: str | None = None,
    is_deletable: Callable[[Any], bool] = lambda _record: True,
    on_delete: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Shared delete-with-report helper.

    Returns the `{deleted, blocked, missing}` shape every service already
    returns. ``is_deletable`` gates deletion per row (training/evaluation
    only delete rows in a terminal status; inference runs have no status and
    default to always-deletable). ``on_delete`` performs any per-row storage
    cleanup before the row is removed from the session.
    """

    query = select(model)
    if ids:
        query = query.where(model.id.in_(ids))
    if project_id:
        query = query.where(or_(model.project_id == project_id, model.project_id.is_(None)))
    records = db.scalars(query).all()
    found = {record.id for record in records}
    deleted = 0
    blocked: list[str] = []
    for record in records:
        if not is_deletable(record):
            blocked.append(record.id)
            continue
        if on_delete is not None:
            on_delete(record)
        db.delete(record)
        deleted += 1
    db.commit()
    return {
        "deleted": deleted,
        "blocked": blocked,
        "missing": [item_id for item_id in (ids or []) if item_id not in found],
    }
