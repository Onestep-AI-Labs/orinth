import argparse
import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--model-option-id", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--target-max-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--hf-model-id", default="")
    parser.add_argument("--hf-cache-dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    labels = load_labels(dataset_root)

    model_option_id = args.model_option_id or "nlp_tfidf_classifier"
    if args.task_type == "text_classification" and model_option_id == "nlp_tfidf_classifier":
        metrics, predictions = train_text_classifier(
            dataset_root, run_dir, labels, max(args.epochs, 1), args.learning_rate
        )
    elif args.task_type == "text_classification" and model_option_id in {
        "nlp_keras_cnn_classifier",
        "nlp_keras_lstm_classifier",
        "nlp_keras_bilstm_classifier",
    }:
        metrics, predictions = train_keras_text_classifier(
            dataset_root=dataset_root,
            run_dir=run_dir,
            labels=labels,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 8),
            vocab_size=max(args.vocab_size, 100),
            model_kind=model_option_id.replace("nlp_keras_", "").replace("_classifier", ""),
        )
    elif args.task_type == "text_classification" and model_option_id == "hf_bert_text_classifier":
        metrics, predictions = train_hf_text_classifier(
            dataset_root=dataset_root,
            run_dir=run_dir,
            labels=labels,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 8),
            model_id=args.hf_model_id or "bert-base-uncased",
            cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
        )
    elif args.task_type == "summarization" and model_option_id == "nlp_extractive_summarizer":
        metrics, predictions = train_summarizer(dataset_root, run_dir, max(args.epochs, 1))
    elif args.task_type == "summarization" and model_option_id == "nlp_keras_seq2seq_summarizer":
        metrics, predictions = train_keras_seq2seq_summarizer(
            dataset_root=dataset_root,
            run_dir=run_dir,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 16),
            target_max_length=max(args.target_max_length, 8),
            vocab_size=max(args.vocab_size, 100),
        )
    elif args.task_type == "summarization" and model_option_id == "hf_bart_summarizer":
        metrics, predictions = train_hf_summarizer(
            dataset_root=dataset_root,
            run_dir=run_dir,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 16),
            target_max_length=max(args.target_max_length, 8),
            model_id=args.hf_model_id or "facebook/bart-base",
            cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
        )
    elif args.task_type == "question_answering" and model_option_id == "nlp_keyword_qa":
        metrics, predictions = train_qa(dataset_root, run_dir, max(args.epochs, 1))
    elif args.task_type == "question_answering" and model_option_id == "hf_bert_question_answering":
        metrics, predictions = train_hf_qa(
            dataset_root=dataset_root,
            run_dir=run_dir,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 64),
            model_id=args.hf_model_id or "bert-base-uncased",
            cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
        )
    else:
        raise ValueError(f"Unsupported NLP task/model option: {args.task_type}/{model_option_id}")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "validation_predictions.json").write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    print(f"NLP training finished. Results saved to: {run_dir}")


def train_text_classifier(
    dataset_root: Path, run_dir: Path, labels: list[str], epochs: int, learning_rate: float
) -> tuple[dict, list[dict]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_text_classification_split(dataset_root, "train")
    valid = load_text_classification_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Text classification training requires labeled train texts")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    train_texts = [item["text"] for item in train]
    train_y = [int(item["class_id"]) for item in train]
    x_train = vectorizer.fit_transform(train_texts)
    if len(set(train_y)) < 2:
        classifier = DummyClassifier(strategy="constant", constant=train_y[0])
    else:
        classifier = LogisticRegression(max_iter=max(epochs, 1) * 100, C=max(learning_rate, 0.0001))
    classifier.fit(x_train, train_y)

    x_valid = vectorizer.transform([item["text"] for item in valid])
    y_true = [int(item["class_id"]) for item in valid]
    y_pred = [int(value) for value in classifier.predict(x_valid)]
    probabilities = classifier.predict_proba(x_valid) if hasattr(classifier, "predict_proba") else None

    with (run_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "classifier": classifier, "labels": labels}, handle)
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics = {
        "classes": labels,
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
        scores = {}
        if probabilities is not None:
            class_values = [int(value) for value in getattr(classifier, "classes_", [])]
            scores = {
                labels[class_id]: float(probabilities[index][column_index])
                for column_index, class_id in enumerate(class_values)
                if 0 <= class_id < len(labels) and column_index < len(probabilities[index])
            }
            scores.update({label: scores.get(label, 0.0) for label in labels})
        predictions.append(
            {
                "item": item["id"],
                "ground_truth": labels[y_true[index]] if y_true[index] < len(labels) else str(y_true[index]),
                "prediction": labels[y_pred[index]] if y_pred[index] < len(labels) else str(y_pred[index]),
                "scores": scores,
            }
        )
    write_results_csv(run_dir / "results.csv", epochs, {"accuracy": accuracy, "macro_f1": macro_f1})
    return metrics, predictions


def train_keras_text_classifier(
    *,
    dataset_root: Path,
    run_dir: Path,
    labels: list[str],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    vocab_size: int,
    model_kind: str,
) -> tuple[dict, list[dict]]:
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_text_classification_split(dataset_root, "train")
    valid = load_text_classification_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Keras text classification training requires labeled train texts")

    train_texts = [item["text"] for item in train]
    valid_texts = [item["text"] for item in valid]
    train_y = np.asarray([int(item["class_id"]) for item in train], dtype="int32")
    valid_y = np.asarray([int(item["class_id"]) for item in valid], dtype="int32")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    x_train = _keras_pad_texts(tf, tokenizer, train_texts, max_length)
    x_valid = _keras_pad_texts(tf, tokenizer, valid_texts, max_length)

    model = _build_keras_text_classifier(
        tf=tf,
        model_kind=model_kind,
        vocab_size=vocab_size,
        max_length=max_length,
        num_labels=len(labels),
        learning_rate=learning_rate,
    )
    history = model.fit(
        x_train,
        train_y,
        validation_data=(x_valid, valid_y),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    model.save(run_dir / "best_model.keras")
    model.save(run_dir / "last_model.keras")
    (run_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")

    probabilities = model.predict(x_valid, verbose=0)
    y_pred = [int(value) for value in np.argmax(probabilities, axis=1)]
    y_true = [int(value) for value in valid_y.tolist()]
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics = {
        "classes": labels,
        "model_kind": model_kind,
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
    metadata = {
        "artifact_type": "keras_text_classifier",
        "model_kind": model_kind,
        "labels": labels,
        "max_length": max_length,
        "vocab_size": vocab_size,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    predictions = []
    for index, item in enumerate(valid):
        scores = {
            labels[class_index]: float(probabilities[index][class_index])
            for class_index in range(min(len(labels), probabilities.shape[1]))
        }
        predictions.append(
            {
                "item": item["id"],
                "ground_truth": labels[y_true[index]] if y_true[index] < len(labels) else str(y_true[index]),
                "prediction": labels[y_pred[index]] if y_pred[index] < len(labels) else str(y_pred[index]),
                "scores": scores,
            }
        )
    write_keras_history_csv(run_dir / "results.csv", history.history)
    return metrics, predictions


def train_keras_seq2seq_summarizer(
    *,
    dataset_root: Path,
    run_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    target_max_length: int,
    vocab_size: int,
) -> tuple[dict, list[dict]]:
    import numpy as np
    import tensorflow as tf

    from app.services.metrics import rouge_scores

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_summary_split(dataset_root, "train")
    valid = load_summary_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Keras seq2seq summarization training requires reference summaries")

    start_token = "startseq"
    end_token = "endseq"
    train_sources = [item["text"] for item in train]
    train_targets = [f"{start_token} {item['summary']} {end_token}" for item in train]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        filters="",
        lower=True,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts([*train_sources, *train_targets])
    encoder_train = _keras_pad_texts(tf, tokenizer, train_sources, max_length)
    target_sequences = tokenizer.texts_to_sequences(train_targets)
    decoder_input = tf.keras.preprocessing.sequence.pad_sequences(
        [sequence[:-1] for sequence in target_sequences],
        maxlen=target_max_length,
        padding="post",
        truncating="post",
    )
    decoder_target = tf.keras.preprocessing.sequence.pad_sequences(
        [sequence[1:] for sequence in target_sequences],
        maxlen=target_max_length,
        padding="post",
        truncating="post",
    )

    model = _build_keras_seq2seq(
        tf=tf,
        vocab_size=vocab_size,
        input_max_length=max_length,
        target_max_length=target_max_length,
        learning_rate=learning_rate,
    )
    history = model.fit(
        [encoder_train, decoder_input],
        np.expand_dims(decoder_target, axis=-1),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    model.save(run_dir / "best_model.keras")
    model.save(run_dir / "last_model.keras")
    (run_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    metadata = {
        "artifact_type": "keras_seq2seq_summarizer",
        "labels": ["summary"],
        "max_length": max_length,
        "target_max_length": target_max_length,
        "vocab_size": vocab_size,
        "start_token": start_token,
        "end_token": end_token,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = []
    predictions = []
    for item in valid:
        prediction = _keras_seq2seq_generate(
            tf=tf,
            model=model,
            tokenizer=tokenizer,
            text=item["text"],
            max_length=max_length,
            target_max_length=target_max_length,
            start_token=start_token,
            end_token=end_token,
        )
        score = rouge_scores(prediction, item["summary"])
        rows.append(score)
        predictions.append(
            {
                "item": item["id"],
                "reference": item["summary"],
                "prediction": prediction,
                "scores": score,
            }
        )
    metrics = average_scores(rows)
    metrics["model_kind"] = "keras_seq2seq"
    write_keras_history_csv(run_dir / "results.csv", history.history)
    return metrics, predictions


def train_summarizer(dataset_root: Path, run_dir: Path, epochs: int) -> tuple[dict, list[dict]]:
    from app.services.metrics import rouge_scores

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_summary_split(dataset_root, "train")
    valid = load_summary_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Summarization training requires reference summaries")
    keywords = most_common_keywords([item["summary"] for item in train])
    (run_dir / "model.json").write_text(json.dumps({"keywords": keywords}, indent=2), encoding="utf-8")
    rows = []
    predictions = []
    for item in valid:
        prediction = extract_summary(item["text"], keywords)
        score = rouge_scores(prediction, item["summary"])
        rows.append(score)
        predictions.append(
            {
                "item": item["id"],
                "reference": item["summary"],
                "prediction": prediction,
                "scores": score,
            }
        )
    metrics = average_scores(rows)
    metrics["keywords"] = keywords
    write_results_csv(run_dir / "results.csv", epochs, metrics)
    return metrics, predictions


def train_qa(dataset_root: Path, run_dir: Path, epochs: int) -> tuple[dict, list[dict]]:
    from app.services.metrics import qa_scores

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_qa_split(dataset_root, "train")
    valid = load_qa_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Question answering training requires question-answer annotations")
    (run_dir / "model.json").write_text(json.dumps({"examples": train[:100]}, indent=2), encoding="utf-8")
    rows = []
    predictions = []
    for item in valid:
        prediction = answer_question(item["text"], item["question"])
        score = qa_scores(prediction, item["answer"])
        rows.append(score)
        predictions.append(
            {
                "item": item["id"],
                "question": item["question"],
                "reference": item["answer"],
                "prediction": prediction,
                "scores": score,
            }
        )
    metrics = average_scores(rows)
    write_results_csv(run_dir / "results.csv", epochs, metrics)
    return metrics, predictions


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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as transformers_logging

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
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
    metrics = average_scores(rows)
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
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, logging as transformers_logging

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
        prediction = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
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
    metrics = average_scores(rows)
    metrics["model_id"] = model_id
    write_metric_rows_csv(run_dir / "results.csv", history)
    return metrics, predictions


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


def most_common_keywords(texts: list[str]) -> list[str]:
    tokens = []
    for text in texts:
        tokens.extend(tokenize(text))
    return [word for word, _count in Counter(tokens).most_common(24)]


def extract_summary(text: str, keywords: list[str]) -> str:
    sentences = sentences_from_text(text)
    if not sentences:
        return ""
    keyword_set = set(keywords)
    return max(sentences, key=lambda sentence: len(set(tokenize(sentence)) & keyword_set))


def answer_question(text: str, question: str) -> str:
    question_tokens = set(tokenize(question))
    sentences = sentences_from_text(text)
    if not sentences:
        return ""
    return max(sentences, key=lambda sentence: len(set(tokenize(sentence)) & question_tokens))


def _keras_pad_texts(tf, tokenizer, texts: list[str], max_length: int):
    sequences = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )


def _build_keras_text_classifier(
    *,
    tf,
    model_kind: str,
    vocab_size: int,
    max_length: int,
    num_labels: int,
    learning_rate: float,
):
    inputs = tf.keras.Input(shape=(max_length,), dtype="int32")
    x = tf.keras.layers.Embedding(vocab_size, 64)(inputs)
    if model_kind == "cnn":
        x = tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    elif model_kind == "lstm":
        x = tf.keras.layers.LSTM(64)(x)
    elif model_kind == "bilstm":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    else:
        raise ValueError(f"Unsupported Keras text model kind: {model_kind}")
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_keras_seq2seq(
    *,
    tf,
    vocab_size: int,
    input_max_length: int,
    target_max_length: int,
    learning_rate: float,
):
    encoder_inputs = tf.keras.Input(shape=(input_max_length,), dtype="int32", name="encoder_input")
    decoder_inputs = tf.keras.Input(shape=(target_max_length,), dtype="int32", name="decoder_input")
    encoder_embedding = tf.keras.layers.Embedding(vocab_size, 96, name="encoder_embedding")(encoder_inputs)
    _encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
        96,
        return_state=True,
        name="encoder_lstm",
    )(encoder_embedding)
    decoder_embedding = tf.keras.layers.Embedding(vocab_size, 96, name="decoder_embedding")(decoder_inputs)
    decoder_outputs = tf.keras.layers.LSTM(
        96,
        return_sequences=True,
        name="decoder_lstm",
    )(decoder_embedding, initial_state=[state_h, state_c])
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", name="token_output")(decoder_outputs)
    model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
    )
    return model


def _keras_seq2seq_generate(
    *,
    tf,
    model,
    tokenizer,
    text: str,
    max_length: int,
    target_max_length: int,
    start_token: str,
    end_token: str,
) -> str:
    source = _keras_pad_texts(tf, tokenizer, [text], max_length)
    start_id = int(tokenizer.word_index.get(start_token, 0))
    end_id = int(tokenizer.word_index.get(end_token, 0))
    decoder = [start_id]
    for _ in range(target_max_length):
        decoder_input = tf.keras.preprocessing.sequence.pad_sequences(
            [decoder],
            maxlen=target_max_length,
            padding="post",
            truncating="post",
        )
        prediction = model.predict([source, decoder_input], verbose=0)[0]
        next_position = min(len(decoder) - 1, prediction.shape[0] - 1)
        next_id = int(prediction[next_position].argmax())
        if next_id in {0, end_id}:
            break
        decoder.append(next_id)
    words = [
        tokenizer.index_word.get(token_id, "")
        for token_id in decoder[1:]
        if token_id not in {0, start_id, end_id}
    ]
    return " ".join(word for word in words if word).strip()


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


def tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", text.lower())


def sentences_from_text(text: str) -> list[str]:
    import re

    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if part.strip()]


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
            row = {"epoch": epoch}
            row.update({key: value for key, value in metrics.items() if key in fieldnames})
            writer.writerow(row)


def write_keras_history_csv(path: Path, history: dict[str, list]) -> None:
    epochs = max((len(values) for values in history.values()), default=0)
    rows = []
    for index in range(epochs):
        row = {"epoch": index + 1}
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


if __name__ == "__main__":
    main()
