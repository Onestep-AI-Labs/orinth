"""Notebook storage, the runtime's presence check, and the runs reader.

Nothing here starts `jupyter-server`. That is deliberate: spawning a real server
in the unit suite would make it slow and would fail on a machine without the
optional extra, and the parts worth pinning down are ours — the directory shape,
the refusal to escape it, and the tolerance for reading a file a kernel is still
appending to.
"""

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import NotebookCreate, NotebookUpdate
from app.services.notebooks import runs as runs_module
from app.services.notebooks.runtime import NotebookRuntime
from app.services.notebooks.service import NotebookService


@pytest.fixture
def service(settings: Settings) -> NotebookService:
    storage = Storage(settings)
    storage.ensure()
    return NotebookService(settings, storage)


def make(service: NotebookService, name: str = "Analysis", **kwargs):
    return service.create(NotebookCreate(name=name, **kwargs))


# --- storage shape ------------------------------------------------------------


def test_a_new_notebook_gets_a_directory_a_manifest_and_a_runnable_file(service):
    notebook = make(service)
    directory = service.directory(notebook.id)

    assert (directory / "manifest.json").is_file()
    assert (directory / "runs").is_dir()

    payload = json.loads((directory / "notebook.ipynb").read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    # The kernelspec must name *our* kernel, not `python3`: a notebook opened on
    # another machine has to ask for the interpreter that has `orinth` on its
    # path, not whatever `python3` happens to resolve to there.
    assert payload["metadata"]["kernelspec"]["name"] == "orinth"
    assert notebook.cell_count == len(payload["cells"])


def test_the_path_handed_to_the_client_is_relative_to_the_server_root(service):
    """An absolute path would leak where the workspace lives and would not
    resolve against `jupyter-server`, which is rooted at `storage/notebooks`."""
    notebook = make(service)
    assert notebook.path == f"{notebook.id}/notebook.ipynb"
    assert not Path(notebook.path).is_absolute()


@pytest.mark.parametrize("hostile", ["../escape", "..", "a/b", "/etc/passwd"])
def test_a_traversing_id_cannot_escape_the_notebook_root(service, hostile):
    """`notebook_id` arrives from a URL segment, which makes it as
    attacker-controlled as an upload filename."""
    with pytest.raises(HTTPException) as failure:
        service.directory(hostile)
    assert failure.value.status_code == 404


def test_listing_is_newest_modified_first(service):
    first = make(service, "first")
    second = make(service, "second")
    service.update(first.id, NotebookUpdate(name="first, touched"))

    listed = [item.id for item in service.list_notebooks()]
    assert listed[0] == first.id
    assert second.id in listed


def test_listing_is_scoped_to_the_project(service):
    mine = make(service, "mine", project_id="project-a")
    make(service, "theirs", project_id="project-b")
    assert [item.id for item in service.list_notebooks("project-a")] == [mine.id]


def test_a_corrupt_notebook_still_lists_and_says_it_is_broken(service):
    """Hiding it would make a broken file look like a deleted one, and the user
    cannot fix what they cannot see."""
    notebook = make(service)
    (service.directory(notebook.id) / "notebook.ipynb").write_text("{ not json", encoding="utf-8")

    listed = service.list_notebooks()
    assert [item.id for item in listed] == [notebook.id]
    assert listed[0].valid is False
    assert listed[0].cell_count == 0


def test_duplicate_copies_the_document_but_not_the_runs(service):
    """Runs record what happened in *that* notebook; carrying them into a copy
    would attribute someone else's numbers to a run that never executed."""
    original = make(service)
    run_dir = service.runs_dir(original.id) / "20260101-000000-abcdef"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"id": "r1", "name": "old"}), encoding="utf-8")

    copy = service.duplicate(original.id)

    assert copy.id != original.id
    assert copy.name.endswith("copy")
    assert copy.cell_count == original.cell_count
    assert list(service.runs_dir(copy.id).iterdir()) == []


def test_delete_removes_the_whole_directory(service):
    notebook = make(service)
    directory = service.directory(notebook.id)
    service.delete(notebook.id)
    assert not directory.exists()
    assert service.list_notebooks() == []


# --- templates ----------------------------------------------------------------


def test_every_shipped_template_is_valid_and_self_describing(service):
    """Templates are the SDK's documentation, so a broken one is a broken doc.

    They carry their own metadata rather than being listed in a second registry
    that can drift from the files.
    """
    templates = service.templates()
    ids = {template.id for template in templates}
    assert {"blank", "dataset-eda", "compare-models", "register-cleaned-dataset"} <= ids
    for template in templates:
        assert template.name
        assert len(template.description) > 20, f"{template.id} needs a real description"


def test_creating_from_a_template_copies_its_cells(service):
    notebook = service.create(
        NotebookCreate(name="From template", template_id="dataset-eda")
    )
    assert notebook.cell_count > 2
    body = (service.directory(notebook.id) / "notebook.ipynb").read_text(encoding="utf-8")
    assert "orinth.datasets.load" in body


def test_an_unknown_template_is_a_404_not_a_blank_notebook(service):
    """Silently falling back to blank would look like the template was empty."""
    with pytest.raises(HTTPException) as failure:
        service.create(NotebookCreate(name="x", template_id="does-not-exist"))
    assert failure.value.status_code == 404


def test_a_template_id_cannot_traverse(service):
    with pytest.raises(HTTPException):
        service.create(NotebookCreate(name="x", template_id="../../../etc/passwd"))


# --- runtime presence ---------------------------------------------------------


def test_the_presence_check_does_not_import_jupyter_server():
    """`find_spec` locates a module without executing it.

    Importing `jupyter_server` to find out whether it is installed would pull
    the Jupyter stack into the API process in the act of checking, which is the
    one thing this whole design exists to avoid.
    """
    import sys

    for name in [key for key in sys.modules if key.startswith("jupyter_server")]:
        del sys.modules[name]

    NotebookRuntime.available()

    assert not any(key.startswith("jupyter_server") for key in sys.modules)


def test_a_missing_extra_reports_an_install_hint_rather_than_an_error(
    settings: Settings, monkeypatch
):
    storage = Storage(settings)
    storage.ensure()
    runtime = NotebookRuntime(settings, storage)
    monkeypatch.setattr(NotebookRuntime, "available", staticmethod(lambda: False))

    status = runtime.status()

    assert status.available is False
    assert status.state == "stopped"
    assert status.error is None
    assert status.install_hint and "notebooks" in status.install_hint


def test_the_notebook_port_band_does_not_overlap_serving(settings: Settings):
    """A llama.cpp server and the notebook gateway allocated the same port would
    be a genuinely confusing failure — one of them simply stops working."""
    notebook_start, notebook_end = settings.notebook_ports
    serving_start, serving_end = settings.serving_ports
    assert notebook_end < serving_start or notebook_start > serving_end


# --- runs reader --------------------------------------------------------------


def write_run(runs_dir: Path, run_id: str, **overrides) -> Path:
    directory = runs_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": run_id,
        "notebook_id": "nb",
        "name": "baseline",
        "status": "completed",
        "metric_names": ["loss"],
        **overrides,
    }
    (directory / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_runs_are_listed_newest_first(service):
    notebook = make(service)
    runs_dir = service.runs_dir(notebook.id)
    write_run(runs_dir, "20260101-000000-aaaaaa")
    write_run(runs_dir, "20260102-000000-bbbbbb")

    listed = runs_module.list_runs(runs_dir, notebook.id)
    assert [run.id for run in listed] == ["20260102-000000-bbbbbb", "20260101-000000-aaaaaa"]


def test_a_half_written_metrics_line_is_skipped_not_raised_on(service):
    """The notebook page charts a run while the cell producing it is still
    going, so reading a partially-appended file is the normal case. The writer
    appends one complete line per point precisely so this can tolerate it."""
    notebook = make(service)
    runs_dir = service.runs_dir(notebook.id)
    directory = write_run(runs_dir, "20260101-000000-aaaaaa")
    (directory / "metrics.jsonl").write_text(
        '{"step": 0, "loss": 1.0}\n{"step": 1, "loss": 0.5}\n{"step": 2, "los',
        encoding="utf-8",
    )

    series = runs_module.get_run(runs_dir, notebook.id, "20260101-000000-aaaaaa")

    assert series is not None
    assert [point["step"] for point in series.points] == [0, 1]


def test_non_numeric_values_are_dropped_from_a_series(service):
    """The chart reads every value as a float; a string in there would break it."""
    notebook = make(service)
    runs_dir = service.runs_dir(notebook.id)
    directory = write_run(runs_dir, "20260101-000000-aaaaaa")
    (directory / "metrics.jsonl").write_text(
        '{"step": 0, "loss": 1.0, "note": "hello"}\n', encoding="utf-8"
    )

    series = runs_module.get_run(runs_dir, notebook.id, "20260101-000000-aaaaaa")
    assert series is not None
    assert series.points == [{"step": 0.0, "loss": 1.0}]


def test_an_artifact_name_cannot_traverse_out_of_its_run(service):
    notebook = make(service)
    runs_dir = service.runs_dir(notebook.id)
    write_run(runs_dir, "20260101-000000-aaaaaa")
    assert (
        runs_module.artifact_path(runs_dir, "20260101-000000-aaaaaa", "../../manifest.json")
        is None
    )


def test_a_missing_run_reads_as_none_rather_than_raising(service):
    notebook = make(service)
    assert runs_module.get_run(service.runs_dir(notebook.id), notebook.id, "nope") is None
