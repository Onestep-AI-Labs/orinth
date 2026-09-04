"""Turning rows in a kernel into a dataset the platform recognises.

The sequence is deliberately indirect: serialize to NDJSON, upload it through
`/datasets/ingest` exactly as a browser would, then hand `/prep/apply` a plan
that states everything rather than asking the agent to infer it.

Going through *apply* rather than writing the layout here is the whole point.
Apply is `_create_layout` → `_update_manifest` → the per-modality item writers →
`process_dataset`, all of which phase 21 validated against fourteen real
datasets. A dataset built by that function satisfies the readiness contract
because it is built by the function readiness was written against. Writing the
directories directly would be a second implementation of the layout, and the
first time the two disagree the symptom is a training run that silently reads
nothing.

Going through *detect + plan* would be the opposite mistake: asking the agent to
re-derive a mapping the caller just stated is how a mapping gets guessed wrong.
"""

import json
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from orinth import _http
from orinth.errors import OrinthError
from orinth.types import DatasetRef

#: Formats that follow from a task when the caller does not name one. Chat is
#: not derivable — an `llm_finetune` dataset is instruction *or* chat and only
#: the rows say which — so it is inferred from the columns instead.
_FORMAT_FOR_TASK = {
    "classification": "image_folder",
    "object_detection": "yolo",
    "segmentation": "yolo",
    "text_classification": "text_folder",
    "summarization": "text_folder",
    "question_answering": "text_folder",
    "llm_finetune": "instruction_jsonl",
}

#: role -> the column name that role defaults to when `columns` omits it.
_DEFAULT_COLUMNS = {
    "text": "text",
    "label": "label",
    "summary": "summary",
    "question": "question",
    "answer": "answer",
    "instruction": "instruction",
    "input": "input",
    "output": "output",
    "messages": "messages",
}

#: Which roles each task actually consumes. Sending a mapping for a role the
#: task ignores is harmless, but deriving labels from the wrong column is not.
_ROLES_FOR_TASK = {
    "text_classification": ("text", "label"),
    "summarization": ("text", "summary"),
    "question_answering": ("text", "question", "answer"),
    "llm_finetune": ("instruction", "input", "output", "messages"),
    "language_modeling": ("text",),
}

TERMINAL = {"ready", "planned", "failed", "cancelled", "draft"}
POLL_SECONDS = 1.0
POLL_TIMEOUT = 900.0


def _rows(data: Any) -> list[dict]:
    """Accept a polars frame, a pandas frame, or any sequence of mappings.

    Duck-typed rather than isinstance-checked, because pandas is not a
    dependency here and importing it to test for it would make it one.
    """
    if hasattr(data, "to_dicts"):  # polars
        return list(data.to_dicts())
    if hasattr(data, "to_dict"):  # pandas
        return list(data.to_dict(orient="records"))
    if isinstance(data, Sequence):
        rows = [dict(entry) for entry in data]
        if not rows:
            raise OrinthError("register() was given no rows.")
        return rows
    raise OrinthError(
        "register() takes a polars DataFrame, a pandas DataFrame, or a sequence of dicts."
    )


def _mapping_for(task_type: str, columns: Mapping[str, str] | None, present: set[str]) -> dict:
    """Build the field mapping, defaulting a role to its own name when it exists.

    Only roles the task consumes are filled. A caller who names a column
    explicitly always wins; the default exists so the common case — a frame whose
    columns are already called `text` and `label` — needs no `columns=` at all.
    """
    supplied = dict(columns or {})
    roles = _ROLES_FOR_TASK.get(task_type, ("text", "label"))
    mapping: dict[str, str] = {}
    for role in roles:
        chosen = supplied.get(role) or (_DEFAULT_COLUMNS[role] if _DEFAULT_COLUMNS[role] in present else None)
        if chosen:
            mapping[role] = chosen
    unknown = [name for name in supplied.values() if name not in present]
    if unknown:
        raise OrinthError(
            f"columns= names {unknown} which are not in the data. Present: {sorted(present)}"
        )
    return mapping


def _format_for(task_type: str, explicit: str | None, mapping: dict, present: set[str]) -> str:
    if explicit:
        return explicit
    if task_type == "llm_finetune":
        # Chat and instruction are the same task and different records; the
        # columns are the only evidence, so they decide.
        if mapping.get("messages") or "messages" in present or "conversations" in present:
            return "chat_jsonl"
        return "instruction_jsonl"
    derived = _FORMAT_FOR_TASK.get(task_type)
    if not derived:
        raise OrinthError(
            f"No default format for task '{task_type}'. Pass format= explicitly."
        )
    return derived


def _labels_for(
    task_type: str, explicit: Sequence[str] | None, rows: list[dict], mapping: dict
) -> list[str]:
    if explicit is not None:
        return [str(label) for label in explicit]
    column = mapping.get("label")
    if not column:
        return []
    seen: dict[str, None] = {}
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            continue
        seen.setdefault(str(value), None)
    # Sorted so two registrations of the same data produce the same label
    # indices; an unsorted set would reorder between runs and silently remap
    # every annotation's `class_id`.
    return sorted(seen)


def register(
    data,
    *,
    name: str,
    task_type: str,
    project_id: str,
    format: str | None = None,  # noqa: A002
    labels: Sequence[str] | None = None,
    columns: Mapping[str, str] | None = None,
    split: Mapping[str, float] | None = None,
    seed: int = 42,
    stratify: bool = True,
    wait: bool = True,
) -> DatasetRef:
    from orinth import datasets as datasets_module

    rows = _rows(data)
    present = {key for row in rows for key in row}
    mapping = _mapping_for(task_type, columns, present)
    resolved_format = _format_for(task_type, format, mapping, present)
    resolved_labels = _labels_for(task_type, labels, rows, mapping)
    ratios = dict(split or {"train": 0.7, "valid": 0.2, "test": 0.1})

    dataset_id = _upload(rows, name=name, project_id=project_id, task_type=task_type)
    plan = _plan(
        task_type=task_type,
        format=resolved_format,
        labels=resolved_labels,
        mapping=mapping,
        ratios=ratios,
        seed=seed,
        stratify=stratify,
        rows=len(rows),
    )

    try:
        _http.post(f"/api/datasets/{dataset_id}/prep/apply", json={"plan": plan})
    except OrinthError as error:
        _http.raise_for_project(error, project_id, task_type)

    if wait:
        _wait(dataset_id)
    return datasets_module.get(dataset_id)


def _upload(rows: list[dict], *, name: str, project_id: str, task_type: str) -> str:
    """Stage the rows exactly as a browser upload would leave them.

    One `data.jsonl`, sent with its relative path, so `_staging/` holds what the
    prep agent's own writers expect to find.
    """
    with tempfile.TemporaryDirectory(prefix="orinth-register-") as scratch:
        path = Path(scratch) / "data.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        with path.open("rb") as handle:
            try:
                dataset = _http.post(
                    "/api/datasets/ingest",
                    files=[("files", ("data.jsonl", handle))],
                    data={"project_id": project_id, "name": name, "relative_paths": ["data.jsonl"]},
                    timeout=_http.UPLOAD_TIMEOUT,
                )
            except OrinthError as error:
                _http.raise_for_project(error, project_id, task_type)
    return dataset["id"]


def _plan(
    *,
    task_type: str,
    format: str,  # noqa: A002
    labels: list[str],
    mapping: dict,
    ratios: dict,
    seed: int,
    stratify: bool,
    rows: int,
) -> dict:
    """A fully-specified plan, marked as having come from a person.

    `source="notebook"` rather than `heuristic`: phase 21's transparency contract
    is explicit that a rule's output is never presented as a model's, and
    presenting an author-written plan as a rule's would be the same
    misattribution in the other direction. Every decision carries
    `source="user"` for the same reason — the Overview tab will show these, and
    it must not claim Orinth worked something out that the user stated.
    """
    decisions = [
        {
            "field": "task_type",
            "value": task_type,
            "source": "user",
            "rationale": "Stated by orinth.datasets.register() in a notebook.",
        },
        {
            "field": "format",
            "value": format,
            "source": "user",
            "rationale": "Stated by the notebook, or derived from the task type.",
        },
    ]
    if labels:
        decisions.append(
            {
                "field": "labels",
                "value": labels,
                "source": "user",
                "rationale": f"{len(labels)} distinct values in the label column.",
            }
        )
    if mapping:
        decisions.append(
            {
                # `DecisionField` calls this `mapping`; the plan field it
                # describes is `field_mapping`. Using the wrong one fails
                # validation on apply, which is where this was caught.
                "field": "mapping",
                "value": mapping,
                "source": "user",
                "rationale": "Columns named by the notebook.",
            }
        )

    return {
        "source": "notebook",
        "confidence": 1.0,
        "task_type": task_type,
        "format": format,
        "labels": labels,
        "field_mapping": mapping,
        "split": {
            "train": float(ratios.get("train", 0.7)),
            "valid": float(ratios.get("valid", 0.2)),
            "test": float(ratios.get("test", 0.1)),
            "seed": int(seed),
            "stratify": bool(stratify),
        },
        "rationale": f"Registered from a notebook: {rows} rows.",
        "decisions": decisions,
        "engine": {"mode": "notebook", "notice": "Plan supplied by a notebook."},
    }


def _wait(dataset_id: str) -> None:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        status = _http.get(f"/api/datasets/{dataset_id}/prep/status") or {}
        if str(status.get("state")) in TERMINAL:
            if status.get("state") == "failed":
                raise OrinthError(status.get("error") or "Preparing the dataset failed.")
            return
        time.sleep(POLL_SECONDS)
    raise OrinthError(
        f"Timed out waiting for '{dataset_id}' to finish preparing. "
        "It is still running; check the Datasets page."
    )
