"""The prep agent, end to end: ingest → detect → plan → apply.

`run()` is the whole agent, synchronously — that is what tests and the desktop
build call, and it is the only path that computes anything. `start()` wraps it
on a thread pool and returns at once, because apply is not fast: a real
15,000-row upload is ~45,000 file creates plus ~45,000 renames, and at 37
seconds a request is past what a dev proxy will hold open. It also runs against
the same thread pool serving the studio's own polling of that dataset.

So this follows what recipes, training, evaluation, and exports already do here:
one small executor per job kind, the work off the request path, and the frontend
polling status instead of holding a socket. What it polls is `metadata.prep`:
`state` for the lifecycle readiness already reads, and `step`/`detail`/`progress`
for the sentence shown while waiting. Those are separate fields on purpose — a
run that is only *reading* the staging directory must not make an already-usable
dataset report itself unusable (see `readiness._PREP_REWRITING`).

What is still deferred is the `data_prep_jobs` *row*: a restart forgets a run in
flight, and its dataset stays `applying` until someone runs prep again. The
manifest is the source of truth for what a dataset *is*; a table is needed for
what a run *was*, and that can land with job history.
"""

import shutil
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, BinaryIO

from app.core.config import Settings
from app.core.defaults import DEFAULT_PROJECT_ID
from app.schemas import DatasetPrepPlan, DatasetPrepStatus, DatasetSummary
from app.services.datasets.prep import apply as apply_module
from app.services.datasets.prep.detect import Detection, detect
from app.services.datasets.prep.plan import analyze, blocking_reason
from app.services.datasets.prep.staging import (
    derived_root,
    resolve_within,
    safe_relative_path,
    staged_count,
    staging_root,
)
from app.services.datasets.prep.transform import refine_with_transform
from app.services.job_runner import submit_job

if TYPE_CHECKING:  # pragma: no cover
    from concurrent.futures import ThreadPoolExecutor

    from app.services.datasets.service import DatasetService
    from app.services.settings import SettingsService


class PrepBusyError(RuntimeError):
    """A prep run is already in flight for this dataset.

    Two runs over one dataset would interleave `_reset_items` with the other's
    writes, so the second is refused rather than queued: the user asked for the
    dataset to be prepared, and it already is being.
    """

#: Provisional task for a draft. The `TaskType` literal stays closed — adding an
#: "unknown" member would ripple into training filters, `_media_dir`, project
#: gating, and every frontend switch. Draft-ness lives in `metadata.prep.state`.
DRAFT_TASK_TYPE = "classification"
DRAFT_FORMAT = "image_folder"

#: Per-file cap on a staged upload. Generous, but a bound: without one a single
#: request can fill the disk.
MAX_STAGED_BYTES = 512 * 1024 * 1024
_COPY_CHUNK = 1024 * 1024


# --- progress wording ---------------------------------------------------------
#
# A run takes tens of seconds and used to report one word — "working". These say
# what is happening *and* what it is happening to, because a file count the user
# recognizes is what makes the difference between a progress readout and a
# spinner. They are deliberately plain sentences, not stage names: "detect" means
# nothing to someone who just dropped a folder in.


def _planning_detail(detection: Detection, *, using_model: bool) -> str:
    files = sum(detection.file_counts.values()) if detection.file_counts else 0
    seen = f"Read {files:,} file{'s' if files != 1 else ''}" if files else "Read your upload"
    if detection.task_type:
        subject = str(detection.task_type).replace("_", " ")
        return f"{seen} — looks like {subject}, checking"
    if using_model:
        return f"{seen} — asking the model what this is"
    return f"{seen} — working out what this is"


def _applying_detail(plan: DatasetPrepPlan) -> str:
    task = str(plan.task_type or "dataset").replace("_", " ")
    return f"Writing {task} items into train, valid and test"


def _done_detail(plan: DatasetPrepPlan, summary: DatasetSummary) -> str:
    task = str(plan.task_type or "dataset").replace("_", " ")
    total = sum(split.item_count for split in summary.splits.values())
    return f"Prepared {total:,} items as {task}"


class DatasetPrepService:
    """Ingest raw files and prepare them, over the existing `DatasetService`."""

    def __init__(
        self,
        settings: Settings,
        dataset_service: "DatasetService",
        settings_service: "SettingsService | None" = None,
        executor: "ThreadPoolExecutor | None" = None,
    ) -> None:
        self.settings = settings
        self.datasets = dataset_service
        self.settings_service = settings_service
        self.executor = executor
        #: Datasets with a run in flight. The manifest state says the same thing,
        #: but it is read from disk and written by the worker — this is what the
        #: *request* checks, under a lock, so two clicks cannot both get through.
        self._running: set[str] = set()
        self._lock = threading.Lock()

    # ---- restart reconciliation ----------------------------------------

    #: States that mean "a worker is on it". None of them survives the process
    #: that was doing the work, which is what makes them reconcilable.
    IN_FLIGHT_STATES = ("detecting", "planning", "applying")

    def reconcile_stale_preps(self) -> int:
        """Fail every prep run left mid-flight by a process that is gone.

        A prep state is a string on a manifest with no liveness behind it, so a
        backend killed during a run leaves `planning` on disk *forever*. That is
        not only a wrong label: `PREP_IN_FLIGHT` in the frontend polls the
        dataset catalog every four seconds while any dataset reads that way, and
        the catalog walks every split of every dataset to build. One interrupted
        run three weeks ago becomes a permanent four-second poll of the most
        expensive read in the app.

        Same posture as `reconcile_stale_jobs` and `reconcile_stale_recipes`,
        and for the same reason: at startup, nothing this process did not start
        is running.
        """
        reconciled = 0
        for location in self.datasets._locations():
            metadata = dict(location.metadata or {})
            prep = dict(metadata.get("prep") or {})
            if prep.get("state") not in self.IN_FLIGHT_STATES:
                continue
            prep["state"] = "failed"
            prep["error"] = "Backend restarted before preparation completed"
            prep["updated_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
            metadata["prep"] = prep
            try:
                self.datasets._update_manifest(location, metadata=metadata)
            except Exception:  # noqa: BLE001 - one unreadable manifest must not stop startup
                continue
            reconciled += 1
        return reconciled

    # ---- ingest --------------------------------------------------------

    def create_draft(self, *, project_id: str = DEFAULT_PROJECT_ID, name: str | None = None) -> str:
        """A dataset shell with no declared type, holding only staged files."""
        dataset_id = self.datasets._new_dataset_id(name or "dataset")
        root = self.datasets.storage.datasets / dataset_id
        staging_root(root).mkdir(parents=True, exist_ok=True)
        self.datasets._create_layout(root, DRAFT_FORMAT, DRAFT_TASK_TYPE, [])
        self.datasets._write_manifest(
            root,
            dataset_id=dataset_id,
            name=name or "Untitled dataset",
            format_name=DRAFT_FORMAT,
            project_id=project_id,
            task_type=DRAFT_TASK_TYPE,
            labels=[],
            metadata={
                "created_from": "ingest",
                "prep": {"state": "draft", "staged_files": 0},
            },
        )
        return dataset_id

    def stage_file(
        self, dataset_id: str, *, stream: BinaryIO, filename: str, relative_path: str | None
    ) -> bool:
        """Write one uploaded file into staging, preserving its relative path.

        Returns False when the path could not be made safe, so the caller can
        report how many files were skipped rather than failing the whole upload
        over one bad name.
        """
        location = self.datasets._location(dataset_id)
        root = staging_root(location.root)
        root.mkdir(parents=True, exist_ok=True)

        relative = safe_relative_path(relative_path, filename)
        if relative is None:
            return False
        destination = resolve_within(root, relative)
        if destination is None:
            return False

        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with destination.open("wb") as handle:
            while True:
                chunk = stream.read(_COPY_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_STAGED_BYTES:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    return False
                handle.write(chunk)

        self._touch_staged_count(dataset_id)
        return True

    def _touch_staged_count(self, dataset_id: str) -> None:
        location = self.datasets._location(dataset_id)
        metadata = dict(location.metadata or {})
        prep = dict(metadata.get("prep") or {})
        prep.setdefault("state", "draft")
        prep["staged_files"] = staged_count(location.root)
        prep["step"] = "staging"
        prep["detail"] = f"Received {prep['staged_files']:,} files"
        metadata["prep"] = prep
        self.datasets._update_manifest(location, metadata=metadata)

    # ---- detect + plan -------------------------------------------------

    def detect_dataset(self, dataset_id: str) -> Detection:
        location = self.datasets._location(dataset_id)
        return detect(staging_root(location.root))

    def plan_for(
        self,
        dataset_id: str,
        *,
        allowed_task_types: list[str] | None = None,
        on_step: "Callable[[str, str, float], None] | None" = None,
    ) -> DatasetPrepPlan:
        """Detect, plan, and — if the plan is still stuck — generate a transform.

        The third stage runs whenever the first two failed to produce a plan that
        can actually yield training examples. That is broader than "no task was
        named": a plan naming a task it has no columns to feed is the worse case,
        because it applies — it writes items with no annotations and reports
        success. Both reach the sandbox, which is not restricted to what
        detection thought the data was.

        Never raises; every stage degrades to the one before it.
        """
        note = on_step or (lambda step, detail, progress: None)
        location = self.datasets._location(dataset_id)

        note("detecting", "Reading the files you uploaded", 0.10)
        detection = self.detect_dataset(dataset_id)

        api_key, model = self._llm_config()
        note("planning", _planning_detail(detection, using_model=bool(api_key)), 0.35)
        plan = analyze(
            detection,
            allowed_task_types=allowed_task_types,
            api_key=api_key,
            model=model,
        )
        # Last run's transform output is dropped before this run decides
        # anything. `_derived/` *replaces* the raw rows in apply, so leaving it
        # in place would have this plan ingest the previous plan's data;
        # `write_derived` rewrites the directory, so a transform that does run
        # below is unaffected.
        shutil.rmtree(derived_root(location.root), ignore_errors=True)

        if plan.task_type and blocking_reason(plan, detection) is None:
            return plan
        note(
            "transforming",
            "These columns are not training examples yet — reshaping them",
            0.55,
        )
        return refine_with_transform(
            plan,
            detection,
            root=location.root,
            api_key=api_key,
            model=model,
            allowed_task_types=allowed_task_types,
        )

    def _llm_config(self) -> tuple[str | None, str | None]:
        """The key and model the user configured for recipes, reused here.

        One place chooses a model. A second default would mean the recipes screen
        and the prep agent could silently disagree about which model is in use.
        """
        if self.settings_service is None:
            return None, None
        try:
            # `SettingsService` exposes only `openrouter_key_configured()`; the key
            # itself is deliberately never returned to a client, so it is read off
            # settings here and passed straight to the provider.
            key = (self.settings_service.settings.openrouter_key or "").strip()
            model = self.settings_service.openrouter_model()
        except Exception:
            return None, None
        return (key or None), (model or None)

    # ---- apply ---------------------------------------------------------

    def apply(self, dataset_id: str, plan: DatasetPrepPlan) -> DatasetSummary:
        return apply_module.apply_plan(self.datasets, dataset_id, plan)

    def run(
        self, dataset_id: str, *, allowed_task_types: list[str] | None = None, auto_apply: bool = True
    ) -> tuple[DatasetSummary, DatasetPrepPlan]:
        """The whole agent in one call.

        `auto_apply=False` stops after planning, leaving the dataset a draft.
        That is the review-gate variant, kept one flag away rather than one
        rewrite away.
        """
        location = self.datasets._location(dataset_id)
        allowed = allowed_task_types

        # `detecting` and `planning` are both read-only over `_staging/`, so the
        # dataset stays exactly as trainable as it was — readiness reports them
        # through `busy` rather than as a state. Only the `applying` write below
        # blocks a verdict.
        def note(step: str, detail: str, progress: float) -> None:
            self._step(dataset_id, "planning", step, detail, progress)

        self._set_state(
            dataset_id,
            "planning",
            step="detecting",
            detail="Reading the files you uploaded",
            progress=0.05,
        )
        try:
            plan = self.plan_for(dataset_id, allowed_task_types=allowed, on_step=note)
        except Exception:
            self._set_state(
                dataset_id,
                "failed",
                step="idle",
                detail="Orinth could not read this data.",
                progress=1.0,
            )
            raise

        if not auto_apply or plan.needs_input or not plan.task_type:
            self._set_state(
                dataset_id,
                "planned",
                plan=plan,
                step="idle",
                detail=plan.needs_input or "Planned, waiting for review.",
                progress=1.0,
            )
            return self.datasets.summary(dataset_id), plan

        self._set_state(
            dataset_id,
            "applying",
            plan=plan,
            step="applying",
            detail=_applying_detail(plan),
            progress=0.70,
        )
        try:
            summary = self.apply(dataset_id, plan)
        except Exception:
            self._set_state(
                dataset_id,
                "failed",
                plan=plan,
                step="idle",
                detail="Orinth could not write the prepared dataset.",
                progress=1.0,
            )
            raise
        self._step(dataset_id, "ready", "done", _done_detail(plan, summary), 1.0)
        del location
        return summary, plan

    # ---- background execution ------------------------------------------

    def start(
        self,
        dataset_id: str,
        *,
        allowed_task_types: list[str] | None = None,
        auto_apply: bool = True,
    ) -> DatasetSummary:
        """Queue a prep run and return the dataset as it stands right now.

        The returned summary already reads `planning`, because the state is
        written here rather than in the worker: a client that polls immediately
        must not see `draft` and conclude nothing happened.

        With no executor configured this runs inline and returns the finished
        dataset. That is the desktop and test path, and it keeps `run()` the
        one place the agent is actually defined.
        """
        if self.executor is None:
            self.run(
                dataset_id, allowed_task_types=allowed_task_types, auto_apply=auto_apply
            )
            return self.datasets.summary(dataset_id)

        with self._lock:
            if dataset_id in self._running:
                raise PrepBusyError("Orinth is already preparing this dataset.")
            self._running.add(dataset_id)

        try:
            self._set_state(
                dataset_id,
                "planning",
                step="detecting",
                detail="Reading the files you uploaded",
                progress=0.05,
            )
        except Exception:
            # Nothing was queued, so the claim has to come back off.
            with self._lock:
                self._running.discard(dataset_id)
            raise

        submit_job(self.executor, self._run_tracked, dataset_id, allowed_task_types, auto_apply)
        return self.datasets.summary(dataset_id)

    def _run_tracked(
        self, dataset_id: str, allowed_task_types: list[str] | None, auto_apply: bool
    ) -> None:
        """The worker: run the agent, and make sure a failure is legible.

        Nothing awaits this, so an exception that escapes is an error the user
        would otherwise experience as a dataset stuck on "preparing" forever.
        """
        try:
            self.run(
                dataset_id, allowed_task_types=allowed_task_types, auto_apply=auto_apply
            )
        except Exception as error:  # noqa: BLE001 - a background run must record why it stopped
            self._record_failure(dataset_id, str(error))
        finally:
            with self._lock:
                self._running.discard(dataset_id)

    def prep_status(self, dataset_id: str) -> DatasetPrepStatus:
        """Just the run state, read off the manifest.

        Deliberately not `summary()`: that walks every split to count items and
        costs seconds on a large dataset, which is exactly what a client polling
        every second and a half must not do. This is one JSON read.
        """
        location = self.datasets._location(dataset_id)
        raw = (location.metadata or {}).get("prep")
        if not isinstance(raw, dict):
            return DatasetPrepStatus(state="ready")
        try:
            return DatasetPrepStatus.model_validate(raw)
        except ValueError:
            return DatasetPrepStatus(state="ready")

    def is_running(self, dataset_id: str) -> bool:
        with self._lock:
            return dataset_id in self._running

    def _record_failure(self, dataset_id: str, message: str) -> None:
        """Put the reason on the manifest, where the studio can show it."""
        try:
            location = self.datasets._location(dataset_id)
            metadata = dict(location.metadata or {})
            prep = dict(metadata.get("prep") or {})
            prep["state"] = "failed"
            prep["error"] = message[:500]
            prep["step"] = "idle"
            prep["detail"] = message[:200]
            prep["progress"] = 1.0
            prep["updated_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
            metadata["prep"] = prep
            self.datasets._update_manifest(location, metadata=metadata)
        except Exception:
            # The dataset was deleted mid-run, or its manifest is unreadable.
            # There is nowhere left to record anything, and raising here would
            # only lose the original error.
            pass

    def undo(self, dataset_id: str) -> DatasetSummary:
        return apply_module.undo(self.datasets, dataset_id)

    def discard_staged(self, dataset_id: str) -> DatasetSummary:
        """Reclaim the raw upload once the user is happy with the result."""
        location = self.datasets._location(dataset_id)
        shutil.rmtree(staging_root(location.root), ignore_errors=True)
        shutil.rmtree(derived_root(location.root), ignore_errors=True)
        self._touch_staged_count(dataset_id)
        return self.datasets.summary(dataset_id)

    def _set_state(
        self,
        dataset_id: str,
        state: str,
        plan: DatasetPrepPlan | None = None,
        *,
        step: str | None = None,
        detail: str | None = None,
        progress: float | None = None,
    ) -> None:
        location = self.datasets._location(dataset_id)
        metadata = dict(location.metadata or {})
        prep = dict(metadata.get("prep") or {})
        prep["state"] = state
        prep["staged_files"] = staged_count(location.root)
        prep["updated_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
        # Whatever went wrong last time is not what is happening now.
        prep.pop("error", None)
        if step is not None:
            prep["step"] = step
        if detail is not None:
            prep["detail"] = detail
        if progress is not None:
            prep["progress"] = progress
        if plan is not None:
            prep["plan"] = plan.model_dump(mode="json")
        metadata["prep"] = prep
        self.datasets._update_manifest(location, metadata=metadata)

    def _step(
        self, dataset_id: str, state: str, step: str, detail: str, progress: float
    ) -> None:
        """Record one stage transition, and never let recording it kill the run.

        The manifest write is a side channel for the progress readout; a run that
        completed but failed to say so is strictly better than one that died
        announcing itself.
        """
        try:
            self._set_state(
                dataset_id, state, step=step, detail=detail, progress=progress
            )
        except Exception:  # noqa: BLE001 - progress is advisory, never fatal
            pass

    def stored_plan(self, dataset_id: str) -> DatasetPrepPlan | None:
        """The plan last applied or proposed, read back off the manifest."""
        location = self.datasets._location(dataset_id)
        raw = (location.metadata or {}).get("prep") or {}
        payload = raw.get("plan") if isinstance(raw, dict) else None
        if not isinstance(payload, dict):
            return None
        try:
            return DatasetPrepPlan.model_validate(payload)
        except ValueError:
            return None
