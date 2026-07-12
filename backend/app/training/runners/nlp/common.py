"""Shared dataset loading, scoring, and CSV writing helpers for NLP runners."""

import csv
import json
from pathlib import Path


def load_labels(dataset_root: Path) -> list[str]:
    manifest = dataset_root / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        labels = [str(label) for label in data.get("labels", []) if str(label).strip()]
        if labels:
            return labels
    return ["positive", "negative", "neutral"]


def load_text_classification_split(dataset_root: Path, split: str) -> list[dict]:
    rows = []
    for text_path, annotations in iter_text_annotations(dataset_root, split):
        annotation = next((item for item in annotations if item.get("kind") == "classification"), None)
        if annotation is None:
            continue
        rows.append({
            "id": text_path.name,
            "text": text_path.read_text(encoding="utf-8", errors="replace"),
            "class_id": int(annotation.get("class_id", 0)),
        })
    return rows


def load_summary_split(dataset_root: Path, split: str) -> list[dict]:
    rows = []
    for text_path, annotations in iter_text_annotations(dataset_root, split):
        annotation = next((item for item in annotations if item.get("kind") == "summary"), None)
        if annotation is None:
            continue
        rows.append({
            "id": text_path.name,
            "text": text_path.read_text(encoding="utf-8", errors="replace"),
            "summary": annotation.get("text") or annotation.get("answer") or "",
        })
    return [row for row in rows if row["summary"]]


def load_qa_split(dataset_root: Path, split: str) -> list[dict]:
    rows = []
    for text_path, annotations in iter_text_annotations(dataset_root, split):
        text = text_path.read_text(encoding="utf-8", errors="replace")
        for annotation in annotations:
            if annotation.get("kind") != "qa":
                continue
            rows.append({
                "id": text_path.name,
                "text": text,
                "question": annotation.get("question") or "",
                "answer": annotation.get("answer") or annotation.get("text") or "",
            })
    return [row for row in rows if row["question"] and row["answer"]]


def iter_text_annotations(dataset_root: Path, split: str):
    text_dir = dataset_root / split / "texts"
    annotation_dir = dataset_root / split / "annotations"
    if not text_dir.exists():
        return
    for text_path in sorted(text_dir.glob("*.txt")):
        annotation_path = annotation_dir / f"{text_path.stem}.json"
        annotations = []
        if annotation_path.exists():
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotations = payload.get("annotations", [])
        yield text_path, annotations


def average_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
        for key in keys
    }


def write_results_csv(path: Path, epochs: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", *[key for key, value in metrics.items() if isinstance(value, (int, float))]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            row: dict[str, float] = {"epoch": epoch}
            row.update({key: value for key, value in metrics.items() if key in fieldnames})
            writer.writerow(row)


def write_keras_history_csv(path: Path, history: dict[str, list]) -> None:
    epochs = max((len(values) for values in history.values()), default=0)
    rows = []
    for index in range(epochs):
        row: dict[str, float] = {"epoch": index + 1}
        for key, values in history.items():
            if index < len(values):
                try:
                    row[key] = float(values[index])
                except (TypeError, ValueError):
                    continue
        rows.append(row)
    write_metric_rows_csv(path, rows)


def write_metric_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("epoch\n", encoding="utf-8")
        return
    fieldnames = ["epoch", *sorted({key for row in rows for key in row if key != "epoch"})]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
