"""Unified single-bar progress mapping (download → load → train → save).

The regression these guard: the download phase used to race a
``5 + len(logs)`` fallback to 99 % while the multi-GB base was still at 43 %,
because progress only mapped the training step count. The whole lifecycle now
maps onto one 0–100 bar, so the reported percent stays monotonic and only
reaches ~100 when the run actually finishes.
"""

from app.services.training.artifacts import (
    _EPOCH_PREP_PERCENT,
    _LLM_BANDS,
    epoch_progress,
    llm_progress,
)


def test_llm_download_percent_maps_into_download_band():
    logs = [
        "[1/4] Dataset ready: 120 train / 20 valid records",
        "[2/4] Downloading base model Qwen/Qwen2.5-1.5B — first run only.",
        "  downloading base model: 43% (1.2 GB/2.9 GB)",
    ]
    percent, label = llm_progress(logs, {})
    lo, hi = _LLM_BANDS["download"]
    assert lo <= percent <= hi
    # 43 % of the way through the download band, not 99 %.
    assert percent < 50
    assert "43%" in label


def test_llm_download_does_not_race_to_99_on_many_log_lines():
    logs = ["[2/4] Downloading base model X"] + [
        f"  downloading base model: {p}% (x GB/y GB)" for p in range(0, 44)
    ]
    percent, _ = llm_progress(logs, {})
    assert percent <= _LLM_BANDS["download"][1]


def test_llm_training_maps_step_fraction_into_train_band():
    logs = ["[3/4] Training (LoRA adapter) — 300 steps. Live loss updates below."]
    percent, label = llm_progress(logs, {"step": 150, "max_steps": 300})
    lo, hi = _LLM_BANDS["train"]
    assert label is None  # caller builds the "step N/M · loss …" line
    assert percent == round(lo + (hi - lo) * 0.5, 2)


def test_llm_phase_ordering_is_monotonic():
    prep = llm_progress(["[1/4] Dataset ready"], {})[0]
    download = llm_progress(
        ["[2/4] Downloading base model X", "  downloading base model: 90% (a/b)"], {}
    )[0]
    load = llm_progress(["[2/4] Downloading base model X", "backend: peft (LoRA, bf16)"], {})[0]
    train = llm_progress(["[3/4] Training"], {"step": 1, "max_steps": 300})[0]
    save = llm_progress(["[4/4] Saving adapter…"], {"step": 300, "max_steps": 300})[0]
    assert prep < download < load < train < save


def test_llm_local_base_skips_download_band():
    percent, label = llm_progress(["[2/4] Using local base model: /models/qwen"], {})
    lo, _ = _LLM_BANDS["load"]
    assert percent >= lo
    assert label == "Loading model weights"


def test_epoch_progress_prep_then_fraction():
    assert epoch_progress(0, 30) == (_EPOCH_PREP_PERCENT, "Preparing training")
    percent, label = epoch_progress(15, 30)
    assert label is None
    assert 5 < percent < 99
