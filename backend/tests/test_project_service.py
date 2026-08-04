from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest
from fastapi import HTTPException

from app.core.defaults import (
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    DEFAULT_PROJECT_TASK_TYPES,
    LEGACY_DEFAULT_PROJECT_NAME,
)
from app.core.database import Base
from app.db.models import Project, TrainingJob
from app.schemas import ProjectCreate, ProjectUpdate
from app.services.projects import ProjectService


def _session(tmp_path, name):
    engine = create_engine(f"sqlite:///{tmp_path / name}")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_project_service_creates_default_and_user_project(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'projects.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    service = ProjectService()

    try:
        projects = service.list_projects(db)
        assert projects[0].id == DEFAULT_PROJECT_ID
        assert projects[0].task_types == DEFAULT_PROJECT_TASK_TYPES

        created = service.create_project(
            db,
            ProjectCreate(
                name="General Images",
                task_types=["classification", "object_detection", "segmentation"],
            ),
        )

        assert created.name == "General Images"
        assert db.get(Project, created.id) is not None

        deleted = service.delete_project(db, created.id)

        assert deleted.deleted == 1
        assert db.get(Project, created.id) is None

        with pytest.raises(HTTPException) as exc:
            service.delete_project(db, DEFAULT_PROJECT_ID)
        assert exc.value.status_code == 409
    finally:
        db.close()


def test_delete_project_reports_specific_blockers(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'projects-blockers.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    service = ProjectService()

    try:
        created = service.create_project(db, ProjectCreate(name="Blocked Project"))
        db.add(TrainingJob(id="job-1", project_id=created.id, model_family="yolo"))
        db.commit()

        with pytest.raises(HTTPException) as exc:
            service.delete_project(db, created.id, dataset_count=2)
        assert exc.value.status_code == 409
        assert exc.value.detail == (
            "This project still has 2 datasets and 1 training job. Delete those first."
        )
    finally:
        db.close()


def test_project_service_backfills_default_nlp_tasks(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'projects-backfill.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    service = ProjectService()
    try:
        db.add(
            Project(
                id=DEFAULT_PROJECT_ID,
                name="Legacy Default",
                task_types=["segmentation", "classification"],
                metadata_json={},
            )
        )
        db.commit()

        default = service.ensure_default(db)

        # Every task type, so a fresh install can start any workflow without
        # first editing project settings. Pre-existing entries keep their order.
        assert default.task_types == [
            "segmentation",
            "classification",
            "object_detection",
            "text_classification",
            "summarization",
            "question_answering",
            "llm_finetune",
            "language_modeling",
        ]
    finally:
        db.close()


def test_ensure_default_renames_the_legacy_starter_project(tmp_path):
    """The starter project was vision-only and named for it; it is not anymore."""
    db = _session(tmp_path, "projects-rename.db")
    service = ProjectService()
    try:
        db.add(
            Project(
                id=DEFAULT_PROJECT_ID,
                name=LEGACY_DEFAULT_PROJECT_NAME,
                task_types=["segmentation"],
                metadata_json={"created_from": "system_default"},
            )
        )
        db.commit()

        assert service.ensure_default(db).name == DEFAULT_PROJECT_NAME
    finally:
        db.close()


def test_ensure_default_keeps_a_name_the_user_chose(tmp_path):
    """Only the untouched legacy name is rewritten — a rename is the user's."""
    db = _session(tmp_path, "projects-keep-name.db")
    service = ProjectService()
    try:
        db.add(
            Project(
                id=DEFAULT_PROJECT_ID,
                name="Dental caries study",
                task_types=["segmentation"],
                metadata_json={"created_from": "system_default"},
            )
        )
        db.commit()

        default = service.ensure_default(db)
        assert default.name == "Dental caries study"
        # Task types are still widened; only the name is left alone.
        assert "llm_finetune" in default.task_types
    finally:
        db.close()


def test_archive_round_trips_through_metadata(tmp_path):
    db = _session(tmp_path, "projects-archive.db")
    service = ProjectService()
    try:
        created = service.create_project(db, ProjectCreate(name="Finished Study"))
        assert created.archived is False

        archived = service.update_project(db, created.id, ProjectUpdate(archived=True))
        assert archived.archived is True
        assert db.get(Project, created.id).metadata_json["archived"] is True

        restored = service.update_project(db, created.id, ProjectUpdate(archived=False))
        assert restored.archived is False
        assert "archived" not in db.get(Project, created.id).metadata_json
    finally:
        db.close()


def test_metadata_replacement_preserves_archived_state(tmp_path):
    """`metadata` replaces the dict wholesale, so a client that omits the
    service-owned `archived` key must not silently un-archive the project."""
    db = _session(tmp_path, "projects-archive-metadata.db")
    service = ProjectService()
    try:
        created = service.create_project(db, ProjectCreate(name="Archived Study"))
        service.update_project(db, created.id, ProjectUpdate(archived=True))

        updated = service.update_project(db, created.id, ProjectUpdate(metadata={"note": "kept"}))

        assert updated.archived is True
        assert updated.metadata["note"] == "kept"

        # An explicit `archived` in the same request still wins.
        explicit = service.update_project(
            db, created.id, ProjectUpdate(metadata={"note": "kept"}, archived=False)
        )
        assert explicit.archived is False
    finally:
        db.close()


def test_default_project_cannot_be_archived(tmp_path):
    db = _session(tmp_path, "projects-archive-default.db")
    service = ProjectService()
    try:
        service.ensure_default(db)
        with pytest.raises(HTTPException) as exc:
            service.update_project(db, DEFAULT_PROJECT_ID, ProjectUpdate(archived=True))
        assert exc.value.status_code == 409
    finally:
        db.close()


def test_project_stats_blockers_match_the_delete_message(tmp_path):
    db = _session(tmp_path, "projects-stats.db")
    service = ProjectService()
    try:
        created = service.create_project(db, ProjectCreate(name="Busy Project"))
        db.add(TrainingJob(id="job-1", project_id=created.id, model_family="yolo"))
        db.commit()

        stats = service.project_stats(db, created.id, dataset_count=2)

        assert stats.datasets == 2
        assert stats.training_jobs == 1
        assert stats.deletable is False
        assert stats.blockers == ["2 datasets", "1 training job"]

        # The danger zone renders `blockers`; delete raises with the same strings.
        with pytest.raises(HTTPException) as exc:
            service.delete_project(db, created.id, dataset_count=2)
        assert exc.value.detail == (
            f"This project still has {' and '.join(stats.blockers)}. Delete those first."
        )
    finally:
        db.close()


def test_project_stats_reports_deletable_when_empty(tmp_path):
    db = _session(tmp_path, "projects-stats-empty.db")
    service = ProjectService()
    try:
        created = service.create_project(db, ProjectCreate(name="Empty Project"))

        stats = service.project_stats(db, created.id)
        assert stats.blockers == []
        assert stats.deletable is True

        # The default project is undeletable for a separate reason and reports so.
        service.ensure_default(db)
        assert service.project_stats(db, DEFAULT_PROJECT_ID).deletable is False
    finally:
        db.close()


def test_project_stats_rejects_unknown_project(tmp_path):
    db = _session(tmp_path, "projects-stats-missing.db")
    service = ProjectService()
    try:
        with pytest.raises(HTTPException) as exc:
            service.project_stats(db, "does-not-exist")
        assert exc.value.status_code == 404
    finally:
        db.close()
