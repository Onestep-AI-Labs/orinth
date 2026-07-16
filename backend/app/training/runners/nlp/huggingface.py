"""Hugging Face Transformers NLP model families: BERT classification/QA, BART summarization."""

import json
import os
from pathlib import Path
from typing import Any

from app.training.runners.nlp.common import (
    average_scores,
    load_qa_split,
    load_summary_split,
    load_text_classification_split,
    write_metric_rows_csv,
)


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
) -> tuple[dict, list[dict]]:
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as transformers_logging

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_text_classification_split(dataset_root, "train")
    valid = load_text_classification_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("BERT text classification training requires labeled train texts")

    transformers_logging.set_verbosity_error()
    print(
        "Hugging Face BERT classifier: loading the base checkpoint with a fresh "
        "classification head for this dataset."
    )
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_inputs = tokenizer(
        [item["text"] for item in train],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    train_y = torch.tensor([int(item["class_id"]) for item in train], dtype=torch.long)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for start in range(0, len(train), batch_size):
            batch = _hf_batch(train_inputs, start, batch_size)
            batch["labels"] = train_y[start : start + batch_size]
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": sum(epoch_losses) / max(len(epoch_losses), 1)})

    model_dir = run_dir / "hf_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "metadata.json").write_text(
        json.dumps({"artifact_type": "hf_text_classifier", "labels": labels, "model_id": model_id}, indent=2),
        encoding="utf-8",
    )
    y_true, y_pred, probabilities = _hf_predict_classification(
        model=model,
        tokenizer=tokenizer,
        texts=[item["text"] for item in valid],
        class_ids=[int(item["class_id"]) for item in valid],
        max_length=max_length,
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics = {
        "classes": labels,
        "model_id": model_id,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
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
) -> tuple[dict, list[dict]]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from app.services.metrics import rouge_scores

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_summary_split(dataset_root, "train")
    valid = load_summary_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("BART summarization training requires reference summaries")

    token = _hf_env_token()
    common_kwargs = _hf_kwargs(cache_dir, token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **common_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sources = [item["text"] for item in train]
    targets = [item["summary"] for item in train]
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for start in range(0, len(train), batch_size):
            batch_sources = sources[start : start + batch_size]
            batch_targets = targets[start : start + batch_size]
            inputs = tokenizer(
                batch_sources,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            target_tokens = tokenizer(
                text_target=batch_targets,
                padding=True,
                truncation=True,
                max_length=target_max_length,
                return_tensors="pt",
            )
            labels = target_tokens["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": sum(epoch_losses) / max(len(epoch_losses), 1)})

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
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=target_max_length, num_beams=2)
        prediction = str(tokenizer.decode(output_ids[0], skip_special_tokens=True))
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
) -> tuple[dict, list[dict]]:
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from transformers import logging as transformers_logging

    from app.services.metrics import qa_scores

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_qa_split(dataset_root, "train")
    valid = load_qa_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("BERT QA training requires question-answer annotations")

    transformers_logging.set_verbosity_error()
    print(
        "Hugging Face BERT QA: loading the base checkpoint with a fresh span head "
        "for this dataset."
    )
    token = _hf_env_token()
    common_kwargs = _hf_kwargs(cache_dir, token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id, **common_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    features = [_qa_feature(tokenizer, item, max_length) for item in train]
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for start in range(0, len(features), batch_size):
            items = features[start : start + batch_size]
            batch = _stack_hf_features([item["inputs"] for item in items])
            batch["start_positions"] = torch.stack([item["start_position"] for item in items])
            batch["end_positions"] = torch.stack([item["end_position"] for item in items])
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": sum(epoch_losses) / max(len(epoch_losses), 1)})

    model_dir = run_dir / "hf_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "metadata.json").write_text(
        json.dumps({"artifact_type": "hf_question_answering", "labels": ["answer"], "model_id": model_id}, indent=2),
        encoding="utf-8",
    )
    model.eval()
    rows = []
    predictions = []
    for item in valid:
        inputs = tokenizer(item["question"], item["text"], return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            start = int(torch.argmax(outputs.start_logits, dim=-1)[0])
            end = int(torch.argmax(outputs.end_logits, dim=-1)[0])
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
    metrics: dict[str, Any] = average_scores(rows)
    metrics["model_id"] = model_id
    write_metric_rows_csv(run_dir / "results.csv", history)
    return metrics, predictions


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


def _hf_batch(encoded: dict, start: int, batch_size: int) -> dict:
    return {key: value[start : start + batch_size] for key, value in encoded.items()}


def _hf_predict_classification(
    *,
    model,
    tokenizer,
    texts: list[str],
    class_ids: list[int],
    max_length: int,
) -> tuple[list[int], list[int], list[list[float]]]:
    import torch

    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).cpu().tolist()
    predictions = [int(max(range(len(row)), key=lambda index: row[index])) for row in probabilities]
    return class_ids, predictions, probabilities


def _qa_feature(tokenizer, item: dict, max_length: int) -> dict:
    import torch

    context = item["text"]
    answer = item["answer"]
    answer_start = context.lower().find(answer.lower())
    answer_end = answer_start + len(answer) if answer_start >= 0 else -1
    encoded = tokenizer(
        item["question"],
        context,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)
    start_position = 0
    end_position = 0
    if answer_start >= 0:
        for index, (start, end) in enumerate(offsets):
            if sequence_ids[index] != 1:
                continue
            if start <= answer_start < end:
                start_position = index
            if start < answer_end <= end:
                end_position = index
                break
    inputs = {key: value.squeeze(0) for key, value in encoded.items()}
    return {
        "inputs": inputs,
        "start_position": torch.tensor(start_position, dtype=torch.long),
        "end_position": torch.tensor(end_position or start_position, dtype=torch.long),
    }


def _stack_hf_features(items: list[dict]) -> dict:
    import torch

    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items]) for key in keys}
