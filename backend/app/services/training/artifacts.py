import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
HF_LOAD_REPORT_RE = re.compile(
    r"^("
    r"\[transformers\].*LOAD REPORT.*|"
    r"Key\s+\|\s+Status.*|"
    r"[-+\s|]+|"
    r".*\|\s+(UNEXPECTED|MISSING)\s*\|?.*|"
    r"Notes:|"
    r"-\s+(UNEXPECTED|MISSING):.*|"
    r".*newly initialized.*|"
    r".*different task/architecture.*"
    r")$"
)


def find_best_model(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "weights" / "best.pt",
        run_dir / "best.pt",
        run_dir / "best_model.keras",
        run_dir / "best_unet_model.keras",
        run_dir / "hf_model",
        run_dir / "adapter",
        run_dir / "model",
        run_dir / "model.pkl",
        run_dir / "model.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in run_dir.rglob("best.pt"):
        return candidate
    for candidate in run_dir.rglob("best_model.keras"):
        return candidate
    for candidate in run_dir.rglob("best_unet_model.keras"):
        return candidate
    for candidate in run_dir.rglob("hf_model"):
        return candidate
    for candidate in run_dir.rglob("model.pkl"):
        return candidate
    for candidate in run_dir.rglob("model.json"):
        return candidate
    return None


def parse_yolo_results(path: Path) -> dict:
    history = parse_training_history(path)
    if not history:
        return {}
    return history[-1]


def parse_training_history(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    history = []
    for row in rows:
        metrics: dict[str, float | int] = {}
        for key, value in row.items():
            normalized = key.strip()
            if value is None or value == "":
                continue
            try:
                parsed = float(value)
            except ValueError:
                continue
            metrics[normalized] = round(parsed, 6)
        epoch_value = metrics.get("epoch")
        if isinstance(epoch_value, float):
            metrics["epoch"] = int(epoch_value)
        if metrics:
            history.append(metrics)
    return history


def collect_training_metrics(run_dir: Path) -> dict:
    metrics = parse_yolo_results(run_dir / "results.csv")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            metrics.update(payload)
    return metrics


def collect_training_curves(run_dir: Path) -> dict:
    curves = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            if payload.get("roc_curves"):
                curves["roc"] = payload["roc_curves"]
            if "macro_auc" in payload:
                curves["macro_auc"] = payload.get("macro_auc")
            if "micro_auc" in payload:
                curves["micro_auc"] = payload.get("micro_auc")
    image_artifacts = {}
    for name in [
        "results",
        "confusion_matrix",
        "confusion_matrix_normalized",
        "F1_curve",
        "P_curve",
        "R_curve",
        "PR_curve",
    ]:
        path = run_dir / f"{name}.png"
        if path.exists():
            image_artifacts[name] = str(path)
    if image_artifacts:
        curves["images"] = image_artifacts
    return curves


def clean_log_line(line: str) -> str:
    clean = ANSI_RE.sub("", line.replace("\r", "\n")).strip()
    compact = " ".join(clean.split())
    if HF_LOAD_REPORT_RE.match(compact):
        return ""
    return compact


def iter_process_lines(stream: TextIO) -> Iterator[str]:
    """Yield output split on either newline or carriage return.

    Hugging Face / tqdm progress bars redraw in place with ``\\r`` and no
    ``\\n``, so a plain ``for line in stream`` (which splits on ``\\n`` only)
    buffers the entire multi-GB download into one chunk and the live log jumps
    straight from "loading" to "done". Treating ``\\r`` as a boundary too
    surfaces each progress redraw as its own line, so download percentage
    streams to the training detail page in real time.
    """
    buffer: list[str] = []
    while True:
        char = stream.read(1)
        if char == "":
            break
        buffer.append(char)
        if char in ("\r", "\n"):
            yield "".join(buffer)
            buffer = []
    if buffer:
        yield "".join(buffer)
