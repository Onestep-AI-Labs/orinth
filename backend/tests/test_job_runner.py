"""Unit tests for the shared job-runner wrapper and list/delete helpers.

These exercise `app.services.job_runner` directly against a real model
(`TrainingJob`) so the tests stay decoupled from any single service's
domain logic while still covering the session/status/failure skeleton and
the list/delete query behavior that training, evaluation, and inference
all rely on.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import TrainingJob
from app.services import job_runner


def _make_job(db_session: Session, **overrides: object) -> TrainingJob:
    defaults: dict[str, object] = {
        "id": "job-1",
        "model_family": "yolo",
        "status": "queued",
    }
    defaults.update(overrides)
    job = TrainingJob(**defaults)
    db_session.add(job)
    db_session.commit()
    return job


def test_run_job_lifecycle_marks_running_then_calls_execute(db_session: Session):
    calls: list[str] = []

    def session_factory() -> Session:
        return db_session

    def on_start(db: Session, job: TrainingJob, started_at: datetime) -> None:
        calls.append("on_start")
        assert job.status == "running"

    def execute(db: Session, job: TrainingJob, started_at: datetime) -> None:
        calls.append("execute")
        assert job.status == "running"
        job.status = "completed"
        db.commit()

    _make_job(db_session)

    job_runner.run_job_lifecycle(
        session_factory,
        TrainingJob,
        "job-1",
        on_start=on_start,
        execute=execute,
    )

    assert calls == ["on_start", "execute"]
    stored = db_session.get(TrainingJob, "job-1")
    assert stored is not None
    assert stored.status == "completed"


def test_run_job_lifecycle_returns_early_when_job_missing(db_session: Session):
    on_start_called = False

    def on_start(db: Session, job: TrainingJob, started_at: datetime) -> None:
        nonlocal on_start_called
        on_start_called = True

    cleanup_calls: list[str] = []

    job_runner.run_job_lifecycle(
        lambda: db_session,
        TrainingJob,
        "missing-job",
        on_start=on_start,
        execute=lambda db, job, started_at: None,
        on_cleanup=cleanup_calls.append,
    )

    assert on_start_called is False
    # Cleanup still runs even though the job row never existed, matching the
    # original per-service `finally` blocks.
    assert cleanup_calls == ["missing-job"]


def test_run_job_lifecycle_persists_failure_on_execute_error(db_session: Session):
    _make_job(db_session)

    def on_start(db: Session, job: TrainingJob, started_at: datetime) -> None:
        pass

    def execute(db: Session, job: TrainingJob, started_at: datetime) -> None:
        raise RuntimeError("boom")

    job_runner.run_job_lifecycle(
        lambda: db_session,
        TrainingJob,
        "job-1",
        on_start=on_start,
        execute=execute,
    )

    stored = db_session.get(TrainingJob, "job-1")
    assert stored is not None
    assert stored.status == "failed"
    assert stored.error == "boom"
    assert stored.artifacts["logs"][-1] == "Failed: boom"
    assert stored.artifacts["progress"]["percent"] == 100
    assert stored.artifacts["progress"]["current_step"] == "Failed"


def test_run_job_lifecycle_persists_failure_when_on_start_raises(db_session: Session):
    _make_job(db_session)

    def on_start(db: Session, job: TrainingJob, started_at: datetime) -> None:
        raise ValueError("bad start")

    job_runner.run_job_lifecycle(
        lambda: db_session,
        TrainingJob,
        "job-1",
        on_start=on_start,
        execute=lambda db, job, started_at: None,
    )

    stored = db_session.get(TrainingJob, "job-1")
    assert stored is not None
    assert stored.status == "failed"
    assert stored.error == "bad start"


def test_run_job_lifecycle_runs_cleanup_on_success_and_failure(db_session: Session):
    _make_job(db_session, id="job-a")
    _make_job(db_session, id="job-b")
    cleanup_calls: list[str] = []

    job_runner.run_job_lifecycle(
        lambda: db_session,
        TrainingJob,
        "job-a",
        on_start=lambda db, job, started_at: None,
        execute=lambda db, job, started_at: None,
        on_cleanup=cleanup_calls.append,
    )

    def failing_execute(db: Session, job: TrainingJob, started_at: datetime) -> None:
        raise RuntimeError("boom")

    job_runner.run_job_lifecycle(
        lambda: db_session,
        TrainingJob,
        "job-b",
        on_start=lambda db, job, started_at: None,
        execute=failing_execute,
        on_cleanup=cleanup_calls.append,
    )

    assert cleanup_calls == ["job-a", "job-b"]


def test_list_jobs_orders_newest_first_and_respects_limit(db_session: Session):
    _make_job(db_session, id="job-1")
    _make_job(db_session, id="job-2")
    _make_job(db_session, id="job-3")

    results = job_runner.list_jobs(db_session, TrainingJob, limit=2)

    assert [job.id for job in results] == ["job-3", "job-2"]


def test_list_jobs_filters_by_project_but_includes_project_none(db_session: Session):
    _make_job(db_session, id="job-a", project_id="alpha")
    _make_job(db_session, id="job-b", project_id="beta")
    _make_job(db_session, id="job-c", project_id=None)

    results = job_runner.list_jobs(db_session, TrainingJob, project_id="alpha")

    assert {job.id for job in results} == {"job-a", "job-c"}


def test_delete_jobs_reports_deleted_blocked_and_missing(db_session: Session):
    _make_job(db_session, id="done", status="completed")
    _make_job(db_session, id="running", status="running")

    report = job_runner.delete_jobs(
        db_session,
        TrainingJob,
        ids=["done", "running", "nonexistent"],
        is_deletable=lambda job: job.status in {"completed", "failed", "canceled"},
    )

    assert report["deleted"] == 1
    assert report["blocked"] == ["running"]
    assert report["missing"] == ["nonexistent"]
    assert db_session.get(TrainingJob, "done") is None
    assert db_session.get(TrainingJob, "running") is not None


def test_delete_jobs_invokes_on_delete_before_removal(db_session: Session):
    _make_job(db_session, id="done", status="completed")
    cleaned_up: list[str] = []

    report = job_runner.delete_jobs(
        db_session,
        TrainingJob,
        ids=["done"],
        on_delete=lambda job: cleaned_up.append(job.id),
    )

    assert report["deleted"] == 1
    assert cleaned_up == ["done"]


def test_delete_jobs_default_is_deletable_allows_all(db_session: Session):
    _make_job(db_session, id="queued-job", status="queued")

    report = job_runner.delete_jobs(db_session, TrainingJob, ids=["queued-job"])

    assert report["deleted"] == 1
    assert report["blocked"] == []
