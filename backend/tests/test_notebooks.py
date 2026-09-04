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
    assert {
        "blank",
        "tour-of-orinth",
        "workspace-report",
        "pipeline-image",
        "pipeline-text",
        "pipeline-llm",
        "upload-your-data",
        "dataset-eda",
        "register-cleaned-dataset",
        "merge-datasets",
        "image-augmentation",
        "detection-review",
        "image-dedupe",
        "text-quality",
        "llm-dataset-shape",
        "text-model-probe",
        "train-classifier",
        "hyperparameter-sweep",
        "platform-training-run",
        "compare-models",
        "evaluate-model",
        "platform-evaluation",
        "error-analysis",
        "batch-inference",
        "recorded-inference",
        "threshold-tuning",
    } <= ids
    for template in templates:
        assert template.name
        assert len(template.description) > 20, f"{template.id} needs a real description"
        assert template.category, f"{template.id} needs a category for the picker"


def test_templates_come_back_in_the_order_the_work_happens(service):
    """Alphabetical would open the picker on "Batch inference", which is the
    last thing anyone does."""
    templates = service.templates()
    categories = [template.category for template in templates]

    assert categories[0] == "Start"
    # Every category is contiguous, so the picker can group by walking the list.
    assert len(set(categories)) == len([
        category for index, category in enumerate(categories)
        if index == 0 or categories[index - 1] != category
    ])
    assert categories.index("Training") < categories.index("Inference")
    # A pipeline is the whole loop, so it sits with the entry points rather than
    # under the one step it happens to start with.
    assert categories.index("Pipelines") < categories.index("Data")


def test_every_category_offers_more_than_one_way_in(service):
    """A tab with one card in it is a tab that did not need to exist."""
    from collections import Counter

    counts = Counter(template.category for template in service.templates())
    thin = {category: count for category, count in counts.items() if count < 2}
    assert not thin, f"these categories need another template: {thin}"


def test_the_sdk_modules_are_each_demonstrated_by_a_template(service):
    """The templates are the SDK's documentation, so a module nothing opens with
    is a module nobody will find. This is the check that keeps that true."""
    import json

    from app.services.notebooks.service import TEMPLATE_DIR

    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in TEMPLATE_DIR.glob("*.ipynb")
    )
    for call in (
        "orinth.datasets.load",
        "orinth.datasets.register",
        "orinth.datasets.upload",
        "orinth.datasets.detect",
        "orinth.models.predictor",
        "orinth.models.parameters",
        "orinth.train.start",
        "orinth.train.wait_for",
        "orinth.evaluate.start",
        "orinth.evaluate.compare",
        "orinth.evaluate.per_item",
        "orinth.inference.predict",
        "orinth.runs.start",
    ):
        assert call in corpus, f"no template shows {call}()"
    # Guard against the corpus check passing on prose alone.
    assert json.loads((TEMPLATE_DIR / "tour-of-orinth.ipynb").read_text())["cells"]


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


# --- refusing cleanly when the runtime is down --------------------------------


def test_a_websocket_to_a_stopped_runtime_closes_instead_of_raising(settings: Settings):
    """An `HTTPException` from a websocket route produces a *malformed* 503.

    Starlette writes the denial response and the exception middleware writes
    another, so the message goes out with two `Content-Length` and two
    `Content-Type` headers. Node's HTTP parser rejects that outright
    (`HPE_UNEXPECTED_CONTENT_LENGTH`), which meant the dev proxy in front of the
    app could not relay the refusal at all — what the user saw was the proxy
    erroring, not the 503. Closing before accept sends a well-formed handshake
    rejection instead, so nothing here may raise and nothing may accept.
    """
    import asyncio

    from app.services.notebooks import proxy as proxy_module

    class FakeWebSocket:
        def __init__(self) -> None:
            self.accepted = False
            self.closed: tuple[int, str] | None = None
            self.headers: dict[str, str] = {}
            self.url = None

        async def accept(self, subprotocol=None):  # noqa: ANN001 - test double
            self.accepted = True

        async def close(self, code=1000, reason=""):  # noqa: ANN001 - test double
            self.closed = (code, reason)

    storage = Storage(settings)
    storage.ensure()
    runtime = NotebookRuntime(settings, storage)
    socket = FakeWebSocket()

    asyncio.run(proxy_module.forward_websocket(runtime, socket, "api/kernels/x/channels", ""))

    assert socket.closed is not None, "a stopped runtime must close the socket, not raise"
    assert not socket.accepted, "refusing after accept would leave a half-open connection"
    assert "not running" in socket.closed[1]


# --- which machine a kernel runs on -------------------------------------------


def test_a_device_choice_reaches_the_kernel_environment():
    """The dropdown has to *do* something. `CUDA_VISIBLE_DEVICES` is the one
    variable both torch and TensorFlow honour at import, so it is what makes a
    CPU choice actually mean CPU rather than a label on a GPU run."""
    from app.services.notebooks.runtime import device_environment

    assert device_environment("cpu")["CUDA_VISIBLE_DEVICES"] == "-1"
    assert device_environment("cuda:1")["CUDA_VISIBLE_DEVICES"] == "1"
    assert device_environment("cuda")["CUDA_VISIBLE_DEVICES"] == "0"
    # `auto` constrains nothing: each framework picks what it can reach, which
    # on a machine where torch sees MPS and TensorFlow does not is the only
    # answer that is right for both.
    assert "CUDA_VISIBLE_DEVICES" not in device_environment("auto")
    assert device_environment("mps") == {"ORINTH_DEVICE": "mps"}


def test_switching_device_under_a_running_runtime_is_refused_by_name(settings: Settings):
    """Silently ignoring it would be a dropdown that appears to work and does
    nothing until the next restart."""
    from app.services.notebooks.runtime import NotebookRuntimeError

    storage = Storage(settings)
    storage.ensure()
    runtime = NotebookRuntime(settings, storage)

    class FakeProcess:
        def poll(self):
            return None

    runtime.process = FakeProcess()  # type: ignore[assignment]
    runtime.device = "cpu"

    with pytest.raises(NotebookRuntimeError) as failure:
        runtime.start("mps")
    assert "already running on 'cpu'" in str(failure.value)


def test_restarting_moves_the_runtime_to_another_machine(settings: Settings, monkeypatch):
    """One call, not stop-then-start from the client: the device only changes
    across a restart, and two calls leave a gap another tab can start the old
    one in."""
    storage = Storage(settings)
    storage.ensure()
    runtime = NotebookRuntime(settings, storage)

    started: list[str] = []
    monkeypatch.setattr(NotebookRuntime, "_spawn", lambda self: None)
    monkeypatch.setattr(NotebookRuntime, "_wait_ready", lambda self: started.append(self.device))
    monkeypatch.setattr(NotebookRuntime, "available", staticmethod(lambda: True))

    runtime.start("cpu")
    assert started == ["cpu"]
    # `start` on a live runtime with a different device is refused; `restart` is
    # the operation that is allowed to change it.
    runtime.restart("auto")
    assert started == ["cpu", "auto"]
    assert runtime.device == "auto"
