import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# ---------------------------------------------------------------------------
# unified progress (one 0–100 bar across the whole run)
# ---------------------------------------------------------------------------
#
# The runner streams phase markers (``[1/4] … [4/4]``) plus a self-reported
# download percentage. The old bar mapped only the training step count onto
# 0–100 and, before that, raced a ``5 + len(logs)`` fallback straight to 99 %
# while the multi-GB download was still at 43 %. These helpers instead map the
# entire lifecycle — data prep → base-model download → weight load → training →
# saving — onto a single bar so 100 % means the run has actually finished.

_DOWNLOAD_PCT_RE = re.compile(r"downloading base model:\s*(\d+(?:\.\d+)?)\s*%")
_PHASE_MARKER_RE = re.compile(r"\[(\d)\s*/\s*4\]")

# Fraction of the unified bar each LLM phase owns. Contiguous and ordered so
# the bar only ever moves forward as the run advances through them.
_LLM_BANDS: dict[str, tuple[float, float]] = {
    "prep": (1.0, 5.0),
    "download": (5.0, 35.0),
    "load": (35.0, 45.0),
    "train": (45.0, 97.0),
    "save": (97.0, 99.0),
}
# Non-LLM (YOLO/Keras/NLP) runs are epoch-denominated: a small prep band before
# the first epoch, then the epoch fraction across the rest of the bar.
_EPOCH_BAND: tuple[float, float] = (5.0, 99.0)
_EPOCH_PREP_PERCENT = 3.0


def _band(bounds: tuple[float, float], fraction: float) -> float:
    lo, hi = bounds
    fraction = max(0.0, min(1.0, fraction))
    return round(lo + (hi - lo) * fraction, 2)


def llm_progress(logs: list[str], metrics: dict) -> tuple[float, str | None]:
    """Unified ``(percent, phase_label)`` for an LLM fine-tuning run.

    ``phase_label`` is ``None`` during the training loop so the caller can build
    its richer "step N/M · loss …" line; every other phase names itself.
    """
    joined = "\n".join(logs[-40:]).lower()
    marker = 0
    for line in logs:
        found = _PHASE_MARKER_RE.search(line)
        if found:
            marker = max(marker, int(found.group(1)))

    step = int(metrics.get("step", 0)) if metrics else 0
    max_steps = int(metrics.get("max_steps", 0)) if metrics else 0

    # [4/4] Saving — the final phase before the process exits (100 % is set on
    # completion in the service, not here).
    if marker >= 4 or "saving" in joined or "finished" in joined:
        return _band(_LLM_BANDS["save"], 0.5), "Saving model"

    # Training — a real step count from results.csv is the strongest signal and
    # survives log truncation (the [3/4] marker can scroll out of the window).
    if step and max_steps:
        return _band(_LLM_BANDS["train"], step / max_steps), None
    if marker >= 3:
        return _LLM_BANDS["train"][0], "Preparing training loop"

    # Weights loaded (or a local base, which never downloads) → load band.
    loaded = (
        "base model ready" in joined
        or "backend:" in joined
        or "using local base model" in joined
    )
    if loaded:
        return _band(_LLM_BANDS["load"], 0.5), "Loading model weights"

    # Download — scale the runner's own percentage into the download band.
    download_pct: float | None = None
    for line in reversed(logs):
        found = _DOWNLOAD_PCT_RE.search(line)
        if found:
            download_pct = float(found.group(1))
            break
    if marker >= 2 or download_pct is not None:
        if download_pct is not None:
            return (
                _band(_LLM_BANDS["download"], download_pct / 100),
                f"Downloading base model — {download_pct:.0f}%",
            )
        return _LLM_BANDS["download"][0], "Downloading base model"

    return _band(_LLM_BANDS["prep"], 0.5), "Preparing dataset"


def epoch_progress(processed: int, total: int | None) -> tuple[float, str | None]:
    """Unified ``(percent, phase_label)`` for an epoch-denominated run."""
    if total and processed:
        return _band(_EPOCH_BAND, processed / total), None
    return _EPOCH_PREP_PERCENT, "Preparing training"



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
