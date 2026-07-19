"""Hugging Face Transformers NLP model families: BERT classification/QA, BART summarization.

The loop here is hand-rolled rather than `transformers.Trainer`, but it follows
the standard fine-tuning recipe: shuffled batches, linear warmup and decay,
weight decay excluding bias and LayerNorm, gradient clipping, per-epoch
validation, and best-checkpoint selection. Each of those matters much more on a
hundred-example dataset than on a large one — without them a fine-tune here
collapses to predicting a single constant class.
"""

import json
import math
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.training.runners.nlp.common import (
    average_scores,
    load_qa_split,
    load_summary_split,
    load_text_classification_split,
    write_metric_rows_csv,
)

WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 42
#: Below this, a validation split is too small for its metric to mean anything,
#: so we top it up from train rather than reporting a number nobody should trust.
MIN_VALIDATION_ITEMS = 5


def train_hf_text_classifier(
    *,
    dataset_root: Path,
    run_dir: Path,
    labels: list[str],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    model_id: str,
    cache_dir: Path | None,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, list[dict]]:
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as transformers_logging

    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    train_pool = load_text_classification_split(dataset_root, "train")
    if not train_pool:
        raise ValueError("BERT text classification training requires labeled train texts")
    train, valid, validation_source = _resolve_validation(
        train_pool,
        load_text_classification_split(dataset_root, "valid"),
        seed=seed,
        stratify_key=lambda item: int(item["class_id"]),
    )

    transformers_logging.set_verbosity_error()
    print(
        f"Hugging Face text classifier ({model_id}): loading the base checkpoint "
        "with a fresh classification head for this dataset."
    )
    print(f"train={len(train)} valid={len(valid)} validation_source={validation_source}")
    token = _hf_env_token()
    common_kwargs = _hf_kwargs(cache_dir, token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        **common_kwargs,
    )
    device = _select_device()
    print(f"training on device: {device}")
    model.to(device)

    optimizer = _build_optimizer(model, learning_rate)
    scheduler = _build_scheduler(optimizer, len(train), batch_size, epochs)
    rng = random.Random(seed)
    history: list[dict] = []
    best_state: dict | None = None
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_indices in _shuffled_batches(len(train), batch_size, rng):
            batch = tokenizer(
                [train[index]["text"] for index in batch_indices],
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            batch["labels"] = torch.tensor(
                [int(train[index]["class_id"]) for index in batch_indices],
                dtype=torch.long,
                device=device,
            )
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))

        y_true, y_pred, probabilities, val_loss = _evaluate_classification(
            model=model,
            tokenizer=tokenizer,
            items=valid,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        val_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        history.append(
            {
                "epoch": epoch,
                "loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                "val_loss": val_loss,
                "val_accuracy": float(accuracy_score(y_true, y_pred)),
                "val_macro_f1": val_macro_f1,
            }
        )
        print(f"epoch {epoch}: {history[-1]}")
        # Rewrite after every epoch: the service derives progress percent and
        # the live curves from this file, so writing it once at the end left
        # the progress bar pinned until the run finished.
        write_metric_rows_csv(run_dir / "results.csv", history)
        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            best_state = _snapshot(model)

    if best_state is not None:
        model.load_state_dict(best_state)
    model_dir = run_dir / "hf_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "metadata.json").write_text(
        json.dumps({"artifact_type": "hf_text_classifier", "labels": labels, "model_id": model_id}, indent=2),
        encoding="utf-8",
    )

    y_true, y_pred, probabilities, _ = _evaluate_classification(
        model=model,
        tokenizer=tokenizer,
        items=valid,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    metrics = {
        "classes": labels,
        "model_id": model_id,
        "validation_source": validation_source,
        "validation_items": len(valid),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    predictions = []
    for index, item in enumerate(valid):
        scores = {
            labels[class_index]: float(probabilities[index][class_index])
            for class_index in range(min(len(labels), len(probabilities[index])))
        }
        predictions.append(
            {
                "item": item["id"],
                "ground_truth": labels[y_true[index]] if y_true[index] < len(labels) else str(y_true[index]),
                "prediction": labels[y_pred[index]] if y_pred[index] < len(labels) else str(y_pred[index]),
                "scores": scores,
            }
        )
    write_metric_rows_csv(run_dir / "results.csv", history)
    return metrics, predictions


def train_hf_summarizer(
    *,
    dataset_root: Path,
    run_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    target_max_length: int,
    model_id: str,
    cache_dir: Path | None,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, list[dict]]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from app.services.metrics import rouge_scores

    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    train_pool = load_summary_split(dataset_root, "train")
    if not train_pool:
        raise ValueError("BART summarization training requires reference summaries")
    train, valid, validation_source = _resolve_validation(
        train_pool, load_summary_split(dataset_root, "valid"), seed=seed
    )

    print(f"train={len(train)} valid={len(valid)} validation_source={validation_source}")
    token = _hf_env_token()
    common_kwargs = _hf_kwargs(cache_dir, token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **common_kwargs)
    device = _select_device()
    print(f"training on device: {device}")
    model.to(device)

    optimizer = _build_optimizer(model, learning_rate)
    scheduler = _build_scheduler(optimizer, len(train), batch_size, epochs)
    rng = random.Random(seed)
    history: list[dict] = []
    best_state: dict | None = None
    best_loss = math.inf

    def encode(items: list[dict], indices: list[int]) -> dict:
        inputs = tokenizer(
            [items[index]["text"] for index in indices],
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        target_tokens = tokenizer(
            text_target=[items[index]["summary"] for index in indices],
            padding="longest",
            truncation=True,
            max_length=target_max_length,
            return_tensors="pt",
        )
        target_ids = target_tokens["input_ids"]
        target_ids[target_ids == tokenizer.pad_token_id] = -100
        batch = {key: value.to(device) for key, value in inputs.items()}
        batch["labels"] = target_ids.to(device)
        return batch

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_indices in _shuffled_batches(len(train), batch_size, rng):
            loss = model(**encode(train, batch_indices)).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))

        # Teacher-forced validation loss each epoch: cheap enough to run every
        # time, unlike beam-search generation, which is left for the final
        # ROUGE report on the selected checkpoint.
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(valid), batch_size):
                indices = list(range(start, min(start + batch_size, len(valid))))
                val_losses.append(float(model(**encode(valid, indices)).loss.detach().cpu()))
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        history.append(
            {
                "epoch": epoch,
                "loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                "val_loss": val_loss,
            }
        )
        print(f"epoch {epoch}: {history[-1]}")
        write_metric_rows_csv(run_dir / "results.csv", history)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = _snapshot(model)

    if best_state is not None:
        model.load_state_dict(best_state)
    model_dir = run_dir / "hf_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "metadata.json").write_text(
        json.dumps({"artifact_type": "hf_summarizer", "labels": ["summary"], "model_id": model_id}, indent=2),
        encoding="utf-8",
    )

    model.eval()
    rows = []
    predictions = []
    for item in valid:
        inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_length=target_max_length, num_beams=4, early_stopping=True
            )
        prediction = str(tokenizer.decode(output_ids[0], skip_special_tokens=True)).strip()
        scores = rouge_scores(prediction, item["summary"])
        rows.append(scores)
        predictions.append(
            {
                "item": item["id"],
                "reference": item["summary"],
                "prediction": prediction,
                "scores": scores,
            }
        )
    metrics: dict[str, Any] = average_scores(rows)
    metrics["model_id"] = model_id
    metrics["validation_source"] = validation_source
    metrics["validation_items"] = len(valid)
    write_metric_rows_csv(run_dir / "results.csv", history)
    return metrics, predictions


def train_hf_qa(
    *,
    dataset_root: Path,
    run_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    model_id: str,
    cache_dir: Path | None,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, list[dict]]:
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from transformers import logging as transformers_logging

    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    train_pool = load_qa_split(dataset_root, "train")
    if not train_pool:
        raise ValueError("BERT QA training requires question-answer annotations")
    train, valid, validation_source = _resolve_validation(
        train_pool, load_qa_split(dataset_root, "valid"), seed=seed
    )

    transformers_logging.set_verbosity_error()
    print(f"Hugging Face QA ({model_id}): fine-tuning an extractive span head.")
    print(f"train={len(train)} valid={len(valid)} validation_source={validation_source}")
    token = _hf_env_token()
    common_kwargs = _hf_kwargs(cache_dir, token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id, **common_kwargs)
    device = _select_device()
    print(f"training on device: {device}")
    model.to(device)

    # An answer that is not a literal span of its context cannot supervise a
    # span head. Training on it anyway pins start/end at position 0, which for
    # `[CLS] question [SEP] context` teaches the model to echo the question.
    features = []
    unlocatable = []
    for item in train:
        feature = _qa_feature(tokenizer, item, max_length)
        if feature is None:
            unlocatable.append(item["id"])
            continue
        features.append(feature)
    if unlocatable:
        print(
            f"WARNING: skipped {len(unlocatable)} of {len(train)} training examples whose "
            f"answer is not a literal span of its context: {unlocatable[:5]}"
        )
    if not features:
        raise ValueError(
            "No QA training example had an answer locatable inside its context. "
            "Answers must appear verbatim in the context text."
        )

    optimizer = _build_optimizer(model, learning_rate)
    scheduler = _build_scheduler(optimizer, len(features), batch_size, epochs)
    rng = random.Random(seed)
    history: list[dict] = []
    best_state: dict | None = None
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_indices in _shuffled_batches(len(features), batch_size, rng):
            items = [features[index] for index in batch_indices]
            batch = _stack_hf_features([item["inputs"] for item in items])
            batch = {key: value.to(device) for key, value in batch.items()}
            batch["start_positions"] = torch.stack([item["start_position"] for item in items]).to(device)
            batch["end_positions"] = torch.stack([item["end_position"] for item in items]).to(device)
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))

        rows, _ = _evaluate_qa(
            model=model, tokenizer=tokenizer, items=valid, max_length=max_length, device=device
        )
        val_f1 = float(average_scores(rows).get("f1", 0.0))
        history.append(
            {
                "epoch": epoch,
                "loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                "val_f1": val_f1,
                "val_exact_match": float(average_scores(rows).get("exact_match", 0.0)),
            }
        )
        print(f"epoch {epoch}: {history[-1]}")
        write_metric_rows_csv(run_dir / "results.csv", history)
        if val_f1 > best_score:
            best_score = val_f1
            best_state = _snapshot(model)

    if best_state is not None:
        model.load_state_dict(best_state)
    model_dir = run_dir / "hf_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "metadata.json").write_text(
        json.dumps({"artifact_type": "hf_question_answering", "labels": ["answer"], "model_id": model_id}, indent=2),
        encoding="utf-8",
    )

    rows, predictions = _evaluate_qa(
        model=model, tokenizer=tokenizer, items=valid, max_length=max_length, device=device
    )
    metrics: dict[str, Any] = average_scores(rows)
    metrics["model_id"] = model_id
    metrics["validation_source"] = validation_source
    metrics["validation_items"] = len(valid)
    metrics["skipped_unlocatable_answers"] = len(unlocatable)
    write_metric_rows_csv(run_dir / "results.csv", history)
    return metrics, predictions


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    """Make a run reproducible.

    Batch shuffling uses its own seeded RNG, but the freshly initialised task
    head (classification and span heads are not carried over from the base
    checkpoint) draws from torch's global RNG. Left unseeded, the same config
    swung between 0.64 and 0.81 macro-F1 across runs, which makes a reported
    metric impossible to compare against another model.
    """
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device():
    """Prefer an accelerator, honouring an explicit override for debugging."""
    import torch

    requested = (os.environ.get("ONESTEP_TRAIN_DEVICE") or "").strip().lower()
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(model, learning_rate: float):
    """AdamW with decay on weights but not on bias or LayerNorm parameters."""
    import torch

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight")
    decay_params = []
    plain_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(marker in name for marker in no_decay):
            plain_params.append(parameter)
        else:
            decay_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": plain_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def _build_scheduler(optimizer, item_count: int, batch_size: int, epochs: int):
    from transformers import get_linear_schedule_with_warmup

    steps_per_epoch = max(1, math.ceil(item_count / max(batch_size, 1)))
    total_steps = max(1, steps_per_epoch * max(epochs, 1))
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )


def _shuffled_batches(count: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    """Batch indices reshuffled every epoch.

    Items arrive in sorted-filename order, which on a hand-built dataset groups
    them by label — so an unshuffled pass yields single-class batches and
    gradients dominated by whichever label sorts first.
    """
    order = list(range(count))
    rng.shuffle(order)
    size = max(batch_size, 1)
    return [order[start : start + size] for start in range(0, len(order), size)]


def _snapshot(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _resolve_validation(
    train: list[dict],
    valid: list[dict],
    *,
    seed: int,
    stratify_key: Callable[[dict], Any] | None = None,
) -> tuple[list[dict], list[dict], str]:
    """Return (train, valid, source), never scoring on data the model trained on.

    The old behaviour was ``valid = load(...) or train``, which silently
    reported training-set accuracy as validation accuracy whenever the valid
    split was empty — the model looked excellent precisely when it had
    memorised. Here a missing or too-small split is topped up with a holdout
    carved out of train instead, and the choice is recorded in the metrics.
    """
    if len(valid) >= MIN_VALIDATION_ITEMS:
        return train, valid, "valid_split"

    holdout_size = max(MIN_VALIDATION_ITEMS - len(valid), math.ceil(len(train) * 0.2))
    holdout_size = min(holdout_size, max(len(train) - 1, 0))
    if holdout_size <= 0:
        # Nothing can be held back without emptying train; report honestly
        # rather than pretending the number means something.
        return train, valid or train, "train_split_reused"

    rng = random.Random(seed)
    if stratify_key is not None:
        grouped: dict[Any, list[int]] = {}
        for index, item in enumerate(train):
            grouped.setdefault(stratify_key(item), []).append(index)
        picked: list[int] = []
        per_group = max(1, holdout_size // max(len(grouped), 1))
        for key in sorted(grouped, key=str):
            indices = grouped[key][:]
            rng.shuffle(indices)
            picked.extend(indices[:per_group])
    else:
        indices = list(range(len(train)))
        rng.shuffle(indices)
        picked = indices[:holdout_size]

    picked_set = set(picked)
    remaining = [item for index, item in enumerate(train) if index not in picked_set]
    holdout = [train[index] for index in sorted(picked_set)]
    return remaining, valid + holdout, "holdout_from_train"


def _hf_kwargs(cache_dir: Path | None, token: str | None) -> dict:
    kwargs = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)
    if token:
        kwargs["token"] = token
    return kwargs


def _hf_env_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = (os.environ.get(key) or "").strip()
        if token:
            return token
    return None


def _evaluate_classification(
    *,
    model,
    tokenizer,
    items: list[dict],
    max_length: int,
    batch_size: int,
    device,
) -> tuple[list[int], list[int], list[list[float]], float]:
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    losses: list[float] = []
    size = max(batch_size, 1)
    with torch.no_grad():
        for start in range(0, len(items), size):
            chunk = items[start : start + size]
            batch = tokenizer(
                [item["text"] for item in chunk],
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            targets = torch.tensor(
                [int(item["class_id"]) for item in chunk], dtype=torch.long, device=device
            )
            outputs = model(**batch, labels=targets)
            losses.append(float(outputs.loss.detach().cpu()))
            chunk_probabilities = torch.softmax(outputs.logits, dim=1).cpu().tolist()
            probabilities.extend(chunk_probabilities)
            y_true.extend(int(item["class_id"]) for item in chunk)
            y_pred.extend(
                int(max(range(len(row)), key=lambda index: row[index])) for row in chunk_probabilities
            )
    model.train()
    return y_true, y_pred, probabilities, sum(losses) / max(len(losses), 1)


def _evaluate_qa(*, model, tokenizer, items: list[dict], max_length: int, device):
    import torch

    from app.services.metrics import qa_scores

    model.eval()
    rows = []
    predictions = []
    with torch.no_grad():
        for item in items:
            inputs = tokenizer(
                item["question"],
                item["text"],
                return_tensors="pt",
                truncation="only_second",
                max_length=max_length,
            )
            sequence_ids = inputs.sequence_ids(0)
            moved = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**moved)
            start_logits = outputs.start_logits[0].cpu()
            end_logits = outputs.end_logits[0].cpu()
            # Restrict the span to context tokens: without this the argmax can
            # land on [CLS] or inside the question and "answer" with the question.
            context_mask = torch.tensor(
                [1.0 if sequence_id == 1 else 0.0 for sequence_id in sequence_ids]
            )
            masked_start = start_logits.masked_fill(context_mask == 0, float("-inf"))
            masked_end = end_logits.masked_fill(context_mask == 0, float("-inf"))
            start = int(torch.argmax(masked_start))
            end = int(torch.argmax(masked_end))
            if end < start:
                end = start
            answer_ids = inputs["input_ids"][0][start : end + 1]
            prediction = str(tokenizer.decode(answer_ids, skip_special_tokens=True)).strip()
            scores = qa_scores(prediction, item["answer"])
            rows.append(scores)
            predictions.append(
                {
                    "item": item["id"],
                    "question": item["question"],
                    "reference": item["answer"],
                    "prediction": prediction,
                    "scores": scores,
                }
            )
    model.train()
    return rows, predictions


def _stack_hf_features(features: list[dict]) -> dict:
    import torch

    return {key: torch.stack([feature[key] for feature in features]) for key in features[0]}


def _qa_feature(tokenizer, item: dict, max_length: int) -> dict | None:
    """Token-space start/end for the answer, or ``None`` if it cannot be located.

    Returning ``None`` rather than falling back to position 0 is the point:
    position 0 is `[CLS]`, and supervising it trains the model to answer with
    the question text.
    """
    import torch

    context = item["text"]
    answer = item["answer"]
    answer_start = context.lower().find(answer.lower())
    if answer_start < 0:
        stripped = answer.strip().rstrip(".,;:")
        answer_start = context.lower().find(stripped.lower())
        if answer_start < 0:
            return None
        answer = stripped
    answer_end = answer_start + len(answer)

    encoded = tokenizer(
        item["question"],
        context,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation="only_second",
        padding="max_length",
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)
    start_position = 0
    end_position = 0
    for index, (start, end) in enumerate(offsets):
        if sequence_ids[index] != 1:
            continue
        if start <= answer_start < end:
            start_position = index
        if start < answer_end <= end:
            end_position = index
            break
    if start_position == 0 or end_position == 0:
        # Truncation pushed the answer out of the window.
        return None

    inputs = {key: value.squeeze(0) for key, value in encoded.items()}
    return {
        "inputs": inputs,
        "start_position": torch.tensor(start_position, dtype=torch.long),
        "end_position": torch.tensor(max(end_position, start_position), dtype=torch.long),
    }
