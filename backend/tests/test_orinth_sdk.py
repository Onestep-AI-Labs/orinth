"""The `orinth` package a notebook imports.

The load-bearing test here is the last one: **`app` must never import
`orinth`**. The SDK may reach into the backend — that is its job — but an arrow
in the other direction would drag the notebook stack into the API process, which
is precisely what spawning kernels as subprocesses exists to prevent. It is an
architectural guarantee, so it is asserted rather than assumed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import DatasetCreate, DatasetPrepPlan
from app.services.datasets import DatasetService
from orinth import _register
from orinth.errors import DatasetBusyError, DatasetTooLargeError, OrinthError
from orinth.types import Readiness


@pytest.fixture
def workspace(settings: Settings, monkeypatch):
    """Point the SDK's cached resolvers at the test workspace."""
    storage = Storage(settings)
    storage.ensure()
    from orinth import _workspace

    _workspace.settings.cache_clear()
    _workspace.storage.cache_clear()
    monkeypatch.setattr(_workspace, "settings", lambda: settings)
    monkeypatch.setattr(_workspace, "storage", lambda: storage)
    return storage


# --- the plan register() hands to apply ---------------------------------------


def test_the_generated_plan_validates_against_the_real_schema():
    """`register()` posts this to `/prep/apply`, so a field name that does not
    exist fails at the far end of an upload. Catching it here is cheaper."""
    plan = _register._plan(
        task_type="text_classification",
        format="text_folder",
        labels=["a", "b"],
        mapping={"text": "body", "label": "y"},
        ratios={"train": 0.7, "valid": 0.2, "test": 0.1},
        seed=7,
        stratify=True,
        rows=100,
    )
    model = DatasetPrepPlan.model_validate(plan)

    assert model.task_type == "text_classification"
    assert model.field_mapping.text == "body"
    assert model.split.seed == 7


def test_a_notebook_plan_is_labelled_as_one():
    """Phase 21's transparency contract says a rule's output is never presented
    as a model's. Presenting a plan the *user wrote* as either would be the same
    misattribution, so `notebook` exists as its own source."""
    plan = _register._plan(
        task_type="summarization",
        format="text_folder",
        labels=[],
        mapping={"text": "t", "summary": "s"},
        ratios={},
        seed=1,
        stratify=False,
        rows=1,
    )
    model = DatasetPrepPlan.model_validate(plan)

    assert model.source == "notebook"
    assert model.engine.mode == "notebook"
    # Every decision is the user's, not something Orinth worked out.
    assert {decision.source for decision in model.decisions} == {"user"}


def test_labels_are_sorted_so_two_registrations_agree_on_class_ids():
    """An unsorted set reorders between runs, which silently remaps every
    annotation's `class_id` — the same data, different labels."""
    rows = [{"y": "cat"}, {"y": "ant"}, {"y": "cat"}, {"y": "bee"}]
    assert _register._labels_for("text_classification", None, rows, {"label": "y"}) == [
        "ant",
        "bee",
        "cat",
    ]


def test_explicit_labels_win_over_derivation():
    rows = [{"y": "cat"}]
    assert _register._labels_for("text_classification", ["x", "y"], rows, {"label": "y"}) == [
        "x",
        "y",
    ]


def test_a_column_that_is_not_in_the_data_is_refused_before_anything_uploads():
    with pytest.raises(OrinthError) as failure:
        _register._mapping_for("text_classification", {"text": "nope"}, {"body", "y"})
    assert "nope" in str(failure.value)
    assert "body" in str(failure.value)


def test_roles_default_to_their_own_name_when_the_column_exists():
    """A frame whose columns are already `text` and `label` needs no `columns=`."""
    assert _register._mapping_for("text_classification", None, {"text", "label"}) == {
        "text": "text",
        "label": "label",
    }


def test_only_the_roles_a_task_consumes_are_mapped():
    mapping = _register._mapping_for("summarization", None, {"text", "summary", "label"})
    assert mapping == {"text": "text", "summary": "summary"}


def test_chat_and_instruction_are_told_apart_by_the_columns():
    """Both are `llm_finetune`; only the rows say which record shape it is."""
    assert _register._format_for("llm_finetune", None, {}, {"messages"}) == "chat_jsonl"
    assert (
        _register._format_for("llm_finetune", None, {}, {"instruction", "output"})
        == "instruction_jsonl"
    )


def test_an_explicit_format_always_wins():
    assert _register._format_for("llm_finetune", "chat_jsonl", {}, {"instruction"}) == "chat_jsonl"


def test_a_task_with_no_derivable_format_says_so_rather_than_guessing():
    with pytest.raises(OrinthError) as failure:
        _register._format_for("something_new", None, {}, set())
    assert "format=" in str(failure.value)


@pytest.mark.parametrize(
    "data",
    [
        [{"a": 1}, {"a": 2}],
        ({"a": 1},),
    ],
)
def test_rows_accepts_any_sequence_of_mappings(data):
    assert _register._rows(data) == [dict(entry) for entry in data]


def test_rows_refuses_something_it_cannot_read():
    with pytest.raises(OrinthError) as failure:
        _register._rows(42)
    assert "DataFrame" in str(failure.value)


def test_rows_reads_a_polars_frame_without_isinstance():
    """Duck-typed on `to_dicts` so pandas never has to be imported to test for
    it — importing a library to check whether it is installed makes it a
    dependency."""
    polars = pytest.importorskip("polars")
    frame = polars.DataFrame([{"a": 1}, {"a": 2}])
    assert _register._rows(frame) == [{"a": 1}, {"a": 2}]


# --- reading ------------------------------------------------------------------


def seed_text_items(service: DatasetService, dataset_id: str, split: str, count: int) -> None:
    """Write `count` text items straight into a split.

    Bypasses the upload route on purpose: what is under test is the SDK's read
    path, and going through the API here would test the API instead.
    """
    location = service._location(dataset_id)
    text_dir = location.root / split / "texts"
    annotation_dir = location.root / split / "annotations"
    text_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        stem = f"row-{index:04d}"
        (text_dir / f"{stem}.txt").write_text(f"row {index}", encoding="utf-8")
        # The sidecar is `{"annotations": [...]}` — `_read_annotation_json`
        # tolerates a bare list too, but only through a `.get` that raises on
        # one, so writing the documented shape is the only safe fixture.
        (annotation_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "class_id": index % 2,
                            "class_name": "a" if index % 2 else "b",
                            "kind": "classification",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )


def test_load_refuses_a_split_over_the_cap_instead_of_truncating(workspace, settings, monkeypatch):
    """A frame that looks complete and is not is the worse failure: every number
    computed from it is quietly wrong."""
    from orinth import datasets as sdk

    service = DatasetService(settings, workspace)
    dataset = service.create_dataset(
        DatasetCreate(name="Big", task_type="text_classification", format="text_folder")
    )
    seed_text_items(service, dataset.id, "train", 12)
    monkeypatch.setattr(settings, "notebook_load_max_rows", 5)
    monkeypatch.setattr(sdk, "_service", lambda: service)

    with pytest.raises(DatasetTooLargeError) as failure:
        sdk.load(dataset.id, "train")
    # The message has to name the ways out, or the cap is just a wall.
    assert "limit=" in str(failure.value)
    assert "records()" in str(failure.value)

    # `limit=` is the documented escape and must actually work.
    assert sdk.load(dataset.id, "train", limit=3).height == 3


def test_load_projects_a_text_split_into_the_documented_columns(workspace, settings, monkeypatch):
    """One return type across every modality is the SDK's core promise; what
    changes per task is the columns."""
    from orinth import datasets as sdk

    service = DatasetService(settings, workspace)
    dataset = service.create_dataset(
        DatasetCreate(name="Texts", task_type="text_classification", format="text_folder")
    )
    seed_text_items(service, dataset.id, "train", 4)
    monkeypatch.setattr(sdk, "_service", lambda: service)

    frame = sdk.load(dataset.id, "train")

    assert frame.height == 4
    assert set(frame.columns) == {"item_id", "split", "text", "label", "label_id"}
    assert frame["split"].to_list() == ["train"] * 4


def test_load_refuses_while_apply_is_rewriting_the_splits(workspace, settings, monkeypatch):
    """Only `applying` drops and repopulates split directories. Detect and plan
    read `_staging/` and leave everything alone — the same distinction readiness
    draws, so the SDK must not be stricter than the badge."""
    from orinth import datasets as sdk

    service = DatasetService(settings, workspace)
    dataset = service.create_dataset(
        DatasetCreate(name="Busy", task_type="text_classification", format="text_folder")
    )
    location = service._location(dataset.id)
    service._update_manifest(location, metadata={"prep": {"state": "applying"}})
    monkeypatch.setattr(sdk, "_service", lambda: service)

    with pytest.raises(DatasetBusyError):
        sdk.load(dataset.id, "train")


@pytest.mark.parametrize("state", ["detecting", "planning", "planned", "ready"])
def test_load_reads_happily_in_every_other_prep_state(workspace, settings, monkeypatch, state):
    from orinth import datasets as sdk

    service = DatasetService(settings, workspace)
    dataset = service.create_dataset(
        DatasetCreate(name=f"State {state}", task_type="text_classification", format="text_folder")
    )
    location = service._location(dataset.id)
    service._update_manifest(location, metadata={"prep": {"state": state}})
    monkeypatch.setattr(sdk, "_service", lambda: service)

    sdk.load(dataset.id, "train")  # must not raise


def test_readiness_is_truthy_when_trainable():
    """`if orinth.datasets.readiness(id):` is the shortest useful check, and it
    should read the way it sounds."""
    assert bool(Readiness(state="ready", trainable=True, summary=""))
    assert not bool(Readiness(state="needs_input", trainable=False, summary=""))


# --- the dependency arrow -----------------------------------------------------

PROBE = """
import sys
import app.main  # the whole application, transitively
leaked = sorted(name for name in sys.modules if name == "orinth" or name.startswith("orinth."))
print("@@leaked@@", ",".join(leaked))
"""


def test_the_backend_never_imports_the_sdk():
    """`orinth` may import `app`; `app` must never import `orinth`.

    An arrow in the other direction would put the notebook stack — and whatever
    a kernel loads through it — inside the API process, which is exactly what
    spawning kernels as subprocesses exists to prevent. Run in a subprocess
    because pytest has already imported both by the time this file executes.
    """
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    line = next(row for row in result.stdout.splitlines() if row.startswith("@@leaked@@"))
    leaked = [name for name in line.split(" ", 1)[1].split(",") if name]

    # `app.cli` reads `orinth.__version__`? It must not — it has its own.
    assert leaked == [], f"app imported the SDK: {leaked}"


def test_the_sdk_exposes_exactly_the_documented_surface():
    """The templates are the SDK's documentation, so the surface has to stay
    small enough to fit in four notebooks. A new public name is a decision."""
    import orinth

    assert sorted(name for name in orinth.__all__ if not name.startswith("_")) == [
        "DatasetBusyError",
        "DatasetRef",
        "DatasetTooLargeError",
        "ModelRef",
        "OrinthError",
        "ProjectRef",
        "ProjectTaskNotAllowed",
        "Readiness",
        "datasets",
        "models",
        "project",
        "projects",
        "runs",
        "settings",
    ]


def test_runs_outside_a_kernel_say_so_rather_than_writing_somewhere_odd(workspace, monkeypatch):
    from orinth import _workspace, runs

    monkeypatch.setattr(_workspace, "notebook_dir", lambda: None)
    with pytest.raises(OrinthError) as failure:
        runs.start("x")
    assert "ORINTH_NOTEBOOK_ID" in str(failure.value)


def test_a_run_writes_one_complete_line_per_point(workspace, monkeypatch, tmp_path):
    """Append-only and complete-per-line is what lets a kernel killed
    mid-experiment leave a readable partial series instead of a truncated blob."""
    from orinth import _workspace, runs

    directory = tmp_path / "nb"
    directory.mkdir()
    monkeypatch.setattr(_workspace, "notebook_dir", lambda: directory)
    monkeypatch.setattr(_workspace, "notebook_id", lambda: "nb")

    with runs.start("baseline", params={"lr": 0.1}) as run:
        run.log(step=0, loss=1.0)
        run.log(step=1, loss=0.5)

    run_dir = next((directory / "runs").iterdir())
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["loss"] for line in lines] == [1.0, 0.5]
    assert json.loads((run_dir / "run.json").read_text())["status"] == "completed"


def test_a_run_that_raised_is_recorded_as_failed(workspace, monkeypatch, tmp_path):
    """An experiment that crashed is a real outcome and must not look the same
    as one nobody finished."""
    from orinth import _workspace, runs

    directory = tmp_path / "nb"
    directory.mkdir()
    monkeypatch.setattr(_workspace, "notebook_dir", lambda: directory)
    monkeypatch.setattr(_workspace, "notebook_id", lambda: "nb")

    with pytest.raises(ValueError), runs.start("boom") as run:
        run.log(step=0, loss=1.0)
        raise ValueError("cell blew up")

    run_dir = next((directory / "runs").iterdir())
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"


def test_log_refuses_a_non_numeric_metric(workspace, monkeypatch, tmp_path):
    """A string in metrics.jsonl breaks every chart that reads it, and the
    caller almost certainly meant `log_text`."""
    from orinth import _workspace, runs

    directory = tmp_path / "nb"
    directory.mkdir()
    monkeypatch.setattr(_workspace, "notebook_dir", lambda: directory)
    monkeypatch.setattr(_workspace, "notebook_id", lambda: "nb")

    run = runs.start("x")
    with pytest.raises(OrinthError) as failure:
        run.log(step=0, note="hello")
    assert "log_text" in str(failure.value)


def test_an_artifact_name_cannot_escape_its_run(workspace, monkeypatch, tmp_path):
    from orinth import _workspace, runs

    directory = tmp_path / "nb"
    directory.mkdir()
    monkeypatch.setattr(_workspace, "notebook_dir", lambda: directory)
    monkeypatch.setattr(_workspace, "notebook_id", lambda: "nb")

    run = runs.start("x")
    written = run.log_artifact(b"data", name="../../escaped.txt")
    assert written.parent.name == "artifacts"
    assert written.name == "escaped.txt"
