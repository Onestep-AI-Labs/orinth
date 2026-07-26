"""Subprocess entrypoint for LLM supervised fine-tuning (phase 14).

Invoked as ``python -m app.training.runners.llm_sft``. Backend selection is
dual: Unsloth 4-bit QLoRA when the device is CUDA and ``unsloth`` imports,
otherwise plain transformers + PEFT LoRA (the MPS/CPU path). Both paths hand
the model to TRL's ``SFTTrainer`` so the loop, metrics, and artifacts are
backend-independent.

The data pipeline is native (patterns re-implemented, no Unsloth Studio code):
``chat_jsonl`` records go through ``tokenizer.apply_chat_template``;
``instruction_jsonl`` records are templated into messages first and then share
the same path. Records are pre-tokenized here so truncation can be counted and
completion masking (loss on assistant tokens only) stays template-agnostic.
"""

import argparse
import csv
import inspect
import json
import math
import os
import random
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from app.ml.llm.catalog import LLM_ADVANCED_PARAMETERS, TARGET_MODULE_OPTIONS
from app.training.runners.advanced import (
    allowed_keys,
    log_ignored,
    parse_advanced,
    partition,
    resolve_precision,
)

LLM_ADVANCED_KEYS = allowed_keys(LLM_ADVANCED_PARAMETERS)

#: Below this an SFT run silently overfits into garbage and wastes an hour
#: doing it, so it fails fast instead.
MIN_TRAIN_RECORDS = 10
SAMPLE_GENERATION_COUNT = 4
SAMPLE_MAX_NEW_TOKENS = 200

# Minimal ChatML-style template applied when a tokenizer ships without one, so
# ``chat_jsonl`` records still train instead of crashing. Logged when used.
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


# ---------------------------------------------------------------------------
# data pipeline (pure helpers — no ML imports, unit-tested directly)
# ---------------------------------------------------------------------------


def load_jsonl_records(dataset_root: Path, split: str) -> list[dict]:
    """Read the phase-10 canonical ``<split>/data.jsonl``; malformed lines are skipped."""
    path = dataset_root / split / "data.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def record_messages(record: dict) -> list[dict] | None:
    """Normalize either record shape into a chat ``messages`` list.

    ``instruction_jsonl`` records are templated into messages (instruction +
    input as the user turn, output as the assistant turn) so both formats share
    one code path through ``apply_chat_template``.
    """
    raw = record.get("messages")
    if isinstance(raw, list) and raw:
        messages = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in raw
            if isinstance(m, dict)
        ]
        return messages or None
    instruction = str(record.get("instruction") or "").strip()
    output = str(record.get("output") or "").strip()
    if not instruction or not output:
        return None
    context = str(record.get("input") or "").strip()
    user_content = f"{instruction}\n\n{context}" if context else instruction
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def resolve_train_valid_records(
    dataset_root: Path,
    *,
    minimum: int = MIN_TRAIN_RECORDS,
    seed: int = 42,
) -> tuple[list[dict], list[dict], str]:
    """Pick the train/valid record pools, tolerating an unsplit dataset.

    Records committed from a recipe (or freshly uploaded) land in the
    ``unassigned`` inbox, so a dataset can hold plenty of data yet have an empty
    ``train`` split. Rather than fail-fast and force a manual split, fall back to
    the ``unassigned`` pool as training data; when no ``valid`` split exists,
    carve a small deterministic holdout from it so the run still reports eval
    loss — but only when that leaves at least ``minimum`` training records.

    Returns ``(train, valid, source)`` where ``source`` is ``"split"`` (a real
    train split was used) or ``"unassigned"`` (the inbox was the training pool).
    """
    train = load_jsonl_records(dataset_root, "train")
    valid = load_jsonl_records(dataset_root, "valid")
    source = "split"
    if not train:
        unassigned = load_jsonl_records(dataset_root, "unassigned")
        if unassigned:
            train = unassigned
            source = "unassigned"
    if train and not valid:
        holdout = max(2, round(len(train) * 0.15))
        if len(train) - holdout >= minimum:
            rng = random.Random(seed)
            order = list(range(len(train)))
            rng.shuffle(order)
            picked = set(order[:holdout])
            valid = [train[index] for index in sorted(picked)]
            train = [record for index, record in enumerate(train) if index not in picked]
    return train, valid, source


def require_min_train_records(count: int, minimum: int = MIN_TRAIN_RECORDS) -> None:
    if count < minimum:
        raise ValueError(
            f"LLM fine-tuning needs at least {minimum} train records, found {count}. "
            "Add more records, or assign them to the train split with the dataset "
            "Splits flow (phase 10), before starting a run."
        )


def ensure_chat_template(tokenizer) -> str | None:
    """Apply a default ChatML-style template when the tokenizer lacks one.

    Returns the log note, or ``None`` when the tokenizer's own template is used.
    """
    if getattr(tokenizer, "chat_template", None):
        return None
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return "Tokenizer has no chat template; applying a default ChatML-style template."


def template_token_ids(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    """Return a flat list of token ids for a chat, across transformers versions.

    ``apply_chat_template(tokenize=True)`` returns a plain list of ids on older
    transformers but a ``BatchEncoding`` dict on newer ones (5.x). Calling
    ``list()`` on the dict yields its *keys* (``["input_ids", "attention_mask"]``,
    length 2), which silently collapsed every example to two tokens and made
    completion masking drop them all. Requesting ``return_dict=True`` and reading
    ``input_ids`` is the shape-stable path; a batched ``[[...]]`` result is
    unwrapped to the single conversation.
    """
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode_messages(
    tokenizer,
    messages: list[dict],
    *,
    max_seq_length: int,
    mask_prompt: bool,
) -> dict[str, Any] | None:
    """Tokenize one chat into ``input_ids``/``labels``, masking prompt tokens.

    Completion masking computes the prompt length by re-rendering everything up
    to the final assistant turn with ``add_generation_prompt=True`` — the same
    prefix the model would see at inference — and sets those label positions to
    -100. Returns ``None`` when the example carries no trainable tokens (e.g.
    truncation removed the whole assistant span).
    """
    full_ids = template_token_ids(tokenizer, messages, add_generation_prompt=False)
    truncated = len(full_ids) > max_seq_length
    if truncated:
        full_ids = full_ids[:max_seq_length]

    labels = list(full_ids)
    if mask_prompt:
        last_assistant = max(
            (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
            default=-1,
        )
        if last_assistant <= 0:
            prompt_length = 0
        else:
            prompt_ids = template_token_ids(
                tokenizer, messages[:last_assistant], add_generation_prompt=True
            )
            prompt_length = len(prompt_ids)
        prompt_length = min(prompt_length, len(labels))
        for index in range(prompt_length):
            labels[index] = -100
        if all(label == -100 for label in labels):
            return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "truncated": truncated,
    }


def prepare_examples(
    tokenizer,
    records: list[dict],
    *,
    max_seq_length: int,
    mask_prompt: bool,
) -> tuple[list[dict], dict[str, int]]:
    """Encode records into training examples, counting what was lost on the way."""
    examples = []
    truncated = 0
    skipped = 0
    for record in records:
        messages = record_messages(record)
        if messages is None:
            skipped += 1
            continue
        encoded = encode_messages(
            tokenizer, messages, max_seq_length=max_seq_length, mask_prompt=mask_prompt
        )
        if encoded is None:
            skipped += 1
            continue
        if encoded.pop("truncated"):
            truncated += 1
        examples.append(encoded)
    return examples, {"truncated": truncated, "skipped": skipped}


def sample_prompts(records: list[dict], count: int = SAMPLE_GENERATION_COUNT) -> list[dict]:
    """Fixed evaluation prompts: prompt messages + reference answer per record."""
    prompts = []
    for record in records:
        messages = record_messages(record)
        if not messages:
            continue
        last_assistant = max(
            (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
            default=-1,
        )
        if last_assistant <= 0:
            continue
        prompts.append(
            {
                "messages": messages[:last_assistant],
                "prompt": next(
                    (m["content"] for m in reversed(messages[:last_assistant]) if m["role"] == "user"),
                    messages[0]["content"],
                ),
                "reference": messages[last_assistant]["content"],
            }
        )
        if len(prompts) >= count:
            break
    return prompts


def filter_supported_kwargs(callable_or_cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the installed TRL/transformers version accepts.

    Argument names have drifted across TRL releases; dropping unknown ones (the
    values here are all optional refinements) beats crashing the run on a
    version skew.
    """
    target = callable_or_cls.__init__ if inspect.isclass(callable_or_cls) else callable_or_cls
    try:
        supported = set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return dict(kwargs)
    if "kwargs" in supported or "args" in supported:
        return dict(kwargs)
    dropped = sorted(key for key in kwargs if key not in supported)
    if dropped:
        print(f"NOTE: this TRL/transformers version does not accept: {', '.join(dropped)}")
    return {key: value for key, value in kwargs.items() if key in supported}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _select_device():
    """cuda > mps > cpu, honouring the ONESTEP_TRAIN_DEVICE override."""
    import torch  # noqa: PLC0415

    requested = (os.environ.get("ONESTEP_TRAIN_DEVICE") or "").strip().lower()
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _hf_env_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = (os.environ.get(key) or "").strip()
        if token:
            return token
    return None


#: The four fine-tuning methods the form offers. LoRA/QLoRA/continued-pretrain
#: all train a PEFT adapter (differing in quantization and loss masking); full
#: fine-tuning updates every weight and produces a standalone model.
FINETUNE_METHODS = ("lora", "qlora", "full", "continued_pretrain")
FINETUNE_METHOD_LABELS = {
    "lora": "LoRA adapter",
    "qlora": "QLoRA (4-bit) adapter",
    "full": "Full fine-tune",
    "continued_pretrain": "Continued pretraining (LoRA)",
}


class _RunConfig:
    """Resolved LLM advanced knobs for one run."""

    def __init__(self, advanced: dict, device_type: str):
        accepted, ignored = partition(advanced, LLM_ADVANCED_KEYS)
        log_ignored(ignored)
        method = str(accepted.get("finetune_method", "lora")).strip().lower()
        if method not in FINETUNE_METHODS:
            print(f"NOTE: unknown fine-tune method {method!r}; using lora.")
            method = "lora"
        self.method = method
        # Full fine-tuning updates all weights (no adapter); the other three
        # attach a PEFT LoRA adapter.
        self.use_peft = method != "full"
        # Continued pretraining trains on the whole sequence (no completion
        # masking) — it is causal-LM continuation, not instruction following.
        self.continued_pretrain = method == "continued_pretrain"
        self.lora_r = int(accepted.get("lora_r", 16))
        self.lora_alpha = int(accepted.get("lora_alpha", 32))
        self.lora_dropout = float(accepted.get("lora_dropout", 0.05))
        modules = accepted.get("target_modules") or TARGET_MODULE_OPTIONS
        self.target_modules = [str(m) for m in modules]
        # QLoRA is the only 4-bit method; the standalone toggle still gates it.
        self.load_in_4bit = method == "qlora" and bool(accepted.get("load_in_4bit", True))
        self.max_seq_length = max(128, int(accepted.get("max_seq_length", 2048)))
        self.packing = bool(accepted.get("packing", False))
        self.lr_scheduler = str(accepted.get("lr_scheduler", "linear"))
        self.warmup_ratio = float(accepted.get("warmup_ratio", 0.03))
        self.weight_decay = float(accepted.get("weight_decay", 0.01))
        self.gradient_accumulation_steps = max(1, int(accepted.get("gradient_accumulation_steps", 4)))
        self.max_steps = max(0, int(accepted.get("max_steps", 0)))
        self.seed = int(accepted.get("seed", 42))
        self.gradient_checkpointing = bool(accepted.get("gradient_checkpointing", True))
        self.train_on_full_sequence = self.continued_pretrain or bool(
            accepted.get("train_on_full_sequence", False)
        )
        precision, note = resolve_precision(
            str(accepted.get("precision", "bf16")),
            cuda=device_type == "cuda",
            mps=device_type == "mps",
        )
        self.precision = precision
        if note:
            print(f"NOTE: {note}")
        if self.load_in_4bit and device_type != "cuda":
            print("NOTE: QLoRA 4-bit requires CUDA + bitsandbytes; training LoRA without quantization.")
            self.load_in_4bit = False
        if not self.use_peft and device_type != "cuda":
            print(
                "NOTE: full fine-tuning updates every weight and is memory-heavy; on "
                f"{device_type.upper()} this may run out of memory — LoRA/QLoRA is recommended."
            )
        print(f"fine-tune method: {FINETUNE_METHOD_LABELS[method]}")


class _PadCollator:
    """Right-pads pre-tokenized examples; label padding is -100 (ignored by the loss)."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch  # noqa: PLC0415

        width = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = width - len(feature["input_ids"])
            batch["input_ids"].append(list(feature["input_ids"]) + [self.pad_token_id] * pad)
            batch["attention_mask"].append(list(feature["attention_mask"]) + [0] * pad)
            batch["labels"].append(list(feature["labels"]) + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _progress_callback(run_dir: Path):
    """Rewrite ``results.csv`` at every logging step (phase-3 progress contract).

    LLM runs are step-denominated: rows carry ``step``/``max_steps`` so the
    service maps them onto processed/total. Writing only at completion would
    leave the progress bar pinned for hours on LLM-scale runs.
    """
    from transformers import TrainerCallback  # noqa: PLC0415

    class _ResultsCsvCallback(TrainerCallback):
        def __init__(self):
            self.rows: list[dict] = []

        def _write(self):
            if not self.rows:
                return
            keys: list[str] = []
            for row in self.rows:
                for key in row:
                    if key not in keys:
                        keys.append(key)
            with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.rows)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs and "eval_loss" not in logs:
                return
            row = {
                "step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": round(float(state.epoch or 0.0), 4),
            }
            if "loss" in logs:
                row["loss"] = round(float(logs["loss"]), 6)
            if "eval_loss" in logs:
                row["val_loss"] = round(float(logs["eval_loss"]), 6)
            # TRL's SFTTrainer reports token-level accuracy (fraction of predicted
            # completion tokens that match) alongside the loss; surface it as the
            # `accuracy`/`val_accuracy` columns so the detail page plots a loss
            # curve on the left and an accuracy curve on the right. Guarded because
            # older TRL/transformers versions do not emit these keys.
            if "mean_token_accuracy" in logs:
                row["accuracy"] = round(float(logs["mean_token_accuracy"]), 6)
            if "eval_mean_token_accuracy" in logs:
                row["val_accuracy"] = round(float(logs["eval_mean_token_accuracy"]), 6)
            if "learning_rate" in logs:
                row["learning_rate"] = logs["learning_rate"]
            # Merge an eval row into the train row for the same step so the
            # curves line up instead of alternating half-empty rows.
            if self.rows and self.rows[-1]["step"] == row["step"]:
                self.rows[-1].update(row)
            else:
                self.rows.append(row)
            self._write()
            print(
                f"step {row['step']}/{row['max_steps']} epoch {row['epoch']}: "
                + " ".join(f"{k}={v}" for k, v in row.items() if k not in {"step", "max_steps", "epoch"})
            )

    return _ResultsCsvCallback()


def _format_size(num_bytes: float) -> str:
    """Human-readable byte size for download progress lines."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _dir_size(path: Path) -> int:
    """Total bytes of files under ``path`` (0 when it does not exist yet)."""
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _prefetch_model(model_ref: str, cache_dir: Path | None, token: str | None) -> None:
    """Pre-download a hub model with reliable, self-reported progress.

    The first-run download of a multi-GB base used to leave the log frozen for
    minutes after "training on device" — Hugging Face's Xet downloader renders
    its own progress that does not reliably reach a piped subprocess. Instead we
    disable Xet and the library bars and report progress ourselves: a background
    thread polls the model's cache folder and prints throttled
    "[2/4] downloading base model: 47% (2.1 GB/4.6 GB)" lines that always stream.
    Populates the same cache ``from_pretrained`` reads (a cache hit follows).
    Local paths are skipped; any failure is non-fatal (the load re-raises the
    authoritative 404/gated/offline error).
    """
    if Path(model_ref).exists():
        return
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: download-progress prefetch unavailable ({exc!r}); loading directly.")
        return

    # Print the framing line FIRST, before any network call — a blocking
    # metadata lookup here used to leave the log silent for the whole download.
    print(
        f"[2/4] Downloading base model {model_ref} — first run only, cached afterwards.",
        flush=True,
    )

    # Disable Xet + the library's own bars so downloads stream to a growing
    # `.incomplete` blob our poller can measure. Set before the download call.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    model_dir = (
        Path(cache_dir) / f"models--{model_ref.replace('/', '--')}" if cache_dir is not None else None
    )
    import threading  # noqa: PLC0415

    stop = threading.Event()
    state: dict[str, int | None] = {"total": None}

    def _report() -> None:
        # Fetch the total size in the background so a slow metadata call never
        # blocks the download or the first progress line.
        try:
            from huggingface_hub import model_info  # noqa: PLC0415

            info = model_info(model_ref, files_metadata=True, token=token)
            state["total"] = sum(f.size for f in (info.siblings or []) if f.size) or None
        except Exception:  # noqa: BLE001 — report raw bytes when the size is unknown
            pass
        while not stop.wait(2.5):
            got = _dir_size(model_dir) if model_dir is not None else 0
            total = state["total"]
            if total:
                percent = min(got, total) / total * 100
                print(
                    f"  downloading base model: {percent:.0f}% "
                    f"({_format_size(min(got, total))}/{_format_size(total)})",
                    flush=True,
                )
            elif got:
                print(f"  downloading base model: {_format_size(got)} so far…", flush=True)
            else:
                print("  downloading base model…", flush=True)

    reporter = threading.Thread(target=_report, daemon=True)
    reporter.start()
    try:
        kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        if token:
            kwargs["token"] = token
        snapshot_download(model_ref, **kwargs)
    except Exception as exc:  # noqa: BLE001 — from_pretrained re-raises the real error
        stop.set()
        print(f"NOTE: download-progress prefetch unavailable ({exc!r}); loading directly.")
        return
    finally:
        stop.set()
        reporter.join(timeout=1.0)
    total = state["total"]
    print(
        f"Base model ready{f' ({_format_size(total)})' if total else ''}, cached for future runs.",
        flush=True,
    )


def _load_model_and_tokenizer(model_ref: str, config: _RunConfig, device, cache_dir: Path | None):
    """Backend selection: Unsloth on CUDA when importable, PEFT LoRA otherwise."""
    token = _hf_env_token()
    common_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        common_kwargs["cache_dir"] = str(cache_dir)
    if token:
        common_kwargs["token"] = token

    # Stream the (first-run) download with progress before the silent load.
    _prefetch_model(model_ref, cache_dir, token)

    # Unsloth's fast path is adapter-only (LoRA/QLoRA); full fine-tuning goes
    # through plain transformers below.
    if config.use_peft and device.type == "cuda" and find_spec("unsloth") is not None:
        try:
            from unsloth import FastLanguageModel  # noqa: PLC0415

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_ref,
                max_seq_length=config.max_seq_length,
                load_in_4bit=config.load_in_4bit,
                **common_kwargs,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
                use_gradient_checkpointing=config.gradient_checkpointing,
                random_state=config.seed,
            )
            print(
                "backend: unsloth"
                + (" (4-bit QLoRA)" if config.load_in_4bit else " (LoRA, no quantization)")
            )
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001 — e.g. CUDA present but bitsandbytes broken
            print(f"NOTE: Unsloth backend failed to initialise ({exc}); falling back to PEFT.")

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_ref, **common_kwargs)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(config.precision, torch.float32)
    quantization_config = None
    if config.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if config.precision == "bf16" else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:  # noqa: BLE001 — bitsandbytes missing/broken → plain LoRA
            print(f"NOTE: 4-bit config unavailable ({exc}); training LoRA without quantization.")
    load_kwargs = dict(common_kwargs)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_ref, dtype=dtype, **load_kwargs)
    if quantization_config is None:
        model.to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if config.use_peft:
            # Required alongside checkpointing for PEFT: without it the frozen
            # base produces no grads into the checkpointed segments.
            model.enable_input_require_grads()

    if not config.use_peft:
        print(f"backend: transformers full fine-tune ({config.precision})")
        return model, tokenizer

    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    if quantization_config is not None:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    label = "4-bit QLoRA" if config.load_in_4bit else "LoRA"
    print(f"backend: peft ({label}, {config.precision})")
    return model, tokenizer


def _generate_samples(model, tokenizer, prompts: list[dict], device) -> list[dict]:
    """Qualitative output from the just-trained adapter for the detail page."""
    import torch  # noqa: PLC0415

    generations = []
    model.eval()
    for item in prompts:
        try:
            # return_dict=True keeps this stable across transformers versions
            # (newer ones return a BatchEncoding rather than a bare tensor).
            encoded = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": SAMPLE_MAX_NEW_TOKENS,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask.to(device)
            with torch.no_grad():
                output = model.generate(input_ids, **generate_kwargs)
            continuation = output[0][input_ids.shape[-1]:]
            generated = str(tokenizer.decode(continuation, skip_special_tokens=True)).strip()
        except Exception as exc:  # noqa: BLE001 — generation must not fail a finished run
            print(f"NOTE: sample generation failed: {exc!r}")
            generated = ""
        generations.append(
            {"prompt": item["prompt"], "reference": item["reference"], "generated": generated}
        )
    return generations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-ref", required=True, help="Hub model id or local model directory")
    parser.add_argument("--model-option-id", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--advanced", default="{}")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    device = _select_device()
    print(f"training on device: {device}")
    advanced = parse_advanced(args.advanced)
    config = _RunConfig(advanced, device.type)

    # Fail fast on data problems before any heavy model download/load. An
    # unsplit dataset (all records in the `unassigned` inbox) still trains: the
    # inbox becomes the training pool with a validation holdout carved out.
    train_records, valid_records, split_source = resolve_train_valid_records(
        dataset_root, seed=config.seed
    )
    if split_source == "unassigned":
        print(
            "No train split found; using the dataset's unassigned records as the "
            "training pool with a small validation holdout."
        )
    require_min_train_records(len(train_records))
    print(f"[1/4] Dataset ready: {len(train_records)} train / {len(valid_records)} valid records")

    import torch  # noqa: PLC0415

    torch.manual_seed(config.seed)

    # The base model download ([2/4]) streams progress from _prefetch_model; for
    # a local base there is nothing to fetch.
    if Path(args.model_ref).exists():
        print(f"[2/4] Using local base model: {args.model_ref}")

    try:
        model, tokenizer = _load_model_and_tokenizer(
            args.model_ref, config, device, Path(args.hf_cache_dir) if args.hf_cache_dir else None
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(f"Out of GPU memory loading {args.model_ref}: {exc}") from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise SystemExit(
                f"Out of memory loading {args.model_ref}. See the catalog memory guidance — "
                "try a 2B-class base, a smaller max_seq_length, or batch size 1."
            ) from exc
        raise

    template_note = ensure_chat_template(tokenizer)
    if template_note:
        print(template_note)

    mask_prompt = not config.train_on_full_sequence
    train_examples, train_stats = prepare_examples(
        tokenizer, train_records, max_seq_length=config.max_seq_length, mask_prompt=mask_prompt
    )
    valid_examples, valid_stats = prepare_examples(
        tokenizer, valid_records, max_seq_length=config.max_seq_length, mask_prompt=mask_prompt
    )
    if not train_examples:
        raise SystemExit(
            "No trainable examples remained after templating/truncation. "
            "Check the dataset records and max_seq_length."
        )
    truncated_count = train_stats["truncated"] + valid_stats["truncated"]
    if truncated_count:
        print(f"{truncated_count} records exceeded max_seq_length={config.max_seq_length} and were truncated")
    if train_stats["skipped"] or valid_stats["skipped"]:
        print(f"skipped {train_stats['skipped'] + valid_stats['skipped']} malformed or fully-masked records")
    if config.packing:
        print("NOTE: packing is not applied by this runner yet; training unpacked.")

    from datasets import Dataset  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(valid_examples) if valid_examples else None

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "trainer"),
        "per_device_train_batch_size": max(1, args.batch_size),
        "per_device_eval_batch_size": max(1, args.batch_size),
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": max(1, args.epochs),
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": config.lr_scheduler,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "epoch" if eval_dataset is not None else "no",
        "seed": config.seed,
        "bf16": config.precision == "bf16",
        "fp16": config.precision == "fp16",
        "gradient_checkpointing": config.gradient_checkpointing,
        "report_to": [],
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "max_steps": config.max_steps if config.max_steps > 0 else -1,
    }
    sft_config = SFTConfig(**filter_supported_kwargs(SFTConfig, sft_kwargs))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": _PadCollator(pad_token_id),
        "processing_class": tokenizer,
        "callbacks": [_progress_callback(run_dir)],
    }
    trainer = SFTTrainer(**filter_supported_kwargs(SFTTrainer, trainer_kwargs))

    steps_note = f"{config.max_steps} steps" if config.max_steps > 0 else f"{max(1, args.epochs)} epochs"
    print(
        f"[3/4] Training ({FINETUNE_METHOD_LABELS[config.method]}) — {steps_note}, "
        f"batch {max(1, args.batch_size)}×{config.gradient_accumulation_steps} accumulation, "
        f"lr {args.learning_rate}. Live loss updates below."
    )

    try:
        result = trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(f"Out of GPU memory during training: {exc}") from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise SystemExit(
                "Out of memory during training. See the catalog memory guidance — try a "
                "2B-class base, a smaller max_seq_length, or batch size 1 with more "
                "gradient accumulation."
            ) from exc
        raise

    # PEFT methods save a small adapter/; full fine-tuning saves a standalone
    # model/ (a complete HF model phase 15 serves without a merge step).
    out_name = "adapter" if config.use_peft else "model"
    artifact_type = "llm_adapter" if config.use_peft else "llm_full"
    print(f"[4/4] Saving {out_name}…")
    out_dir = run_dir / out_name
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Carry the base's SentencePiece vocab so a later GGUF export (Gemma/Llama)
    # keeps the correct converter path instead of failing on the gpt2 fallback.
    try:
        from app.ml.llm.gguf_tools import ensure_sentencepiece_vocab  # noqa: PLC0415

        ensure_sentencepiece_vocab(
            out_dir, args.model_ref, cache_dir=args.hf_cache_dir or None, token=_hf_env_token()
        )
    except Exception as exc:  # noqa: BLE001 — never fail a finished run over a vocab copy
        print(f"NOTE: could not copy SentencePiece vocab ({exc!r}).")

    final_train_loss = float(getattr(result, "training_loss", 0.0) or 0.0)
    metrics: dict[str, Any] = {
        "train_loss": round(final_train_loss, 6),
        "backend": "unsloth" if model.__class__.__module__.startswith("unsloth") else "peft",
        "finetune_method": config.method,
        "model_ref": args.model_ref,
        "truncated_records": truncated_count,
        "train_records": len(train_examples),
        "valid_records": len(valid_examples),
    }
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        val_loss = float(eval_metrics.get("eval_loss", 0.0))
        metrics["val_loss"] = round(val_loss, 6)
        metrics["perplexity"] = round(math.exp(min(val_loss, 20.0)), 4)
        if "eval_mean_token_accuracy" in eval_metrics:
            metrics["val_accuracy"] = round(float(eval_metrics["eval_mean_token_accuracy"]), 6)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    prompts = sample_prompts(valid_records if valid_records else train_records)
    generations = _generate_samples(model, tokenizer, prompts, device) if prompts else []
    (run_dir / "sample_generations.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": artifact_type,
                "finetune_method": config.method,
                "base_model_ref": args.model_ref,
                "model_option_id": args.model_option_id,
                "backend": metrics["backend"],
                "max_seq_length": config.max_seq_length,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "chat_template_source": "default_chatml" if template_note else "tokenizer",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"LLM SFT finished ({FINETUNE_METHOD_LABELS[config.method]}). Saved to: {out_dir}")


if __name__ == "__main__":
    main()
