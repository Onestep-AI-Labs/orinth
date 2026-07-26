"""Non-neural NLP baselines: TF-IDF classifier, keyword extractive summarizer, keyword QA."""

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from app.ml.nlp.baseline.catalog import SKLEARN_ADVANCED_PARAMETERS
from app.training.runners.advanced import allowed_keys, log_ignored, partition
from app.training.runners.nlp.common import (
    average_scores,
    load_qa_split,
    load_summary_split,
    load_text_classification_split,
    write_results_csv,
)

SKLEARN_ADVANCED_KEYS = allowed_keys(SKLEARN_ADVANCED_PARAMETERS)


def parse_ngram_range(value: Any, default: tuple[int, int] = (1, 2)) -> tuple[int, int]:
    """Parse an ``"lower,upper"`` n-gram range spec, falling back on bad input."""

    try:
        lower, upper = (int(part) for part in str(value).split(","))
    except (ValueError, TypeError):
        return default
    if lower < 1 or upper < lower:
        return default
    return lower, upper


def tfidf_classifier_settings(
    advanced: dict[str, Any], epochs: int, learning_rate: float
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Resolve TF-IDF vectorizer and LogisticRegression kwargs from advanced values.

    Returns ``(vectorizer_kwargs, classifier_kwargs, ignored_keys)``. Basic
    ``epochs``/``learning_rate`` seed the defaults; advanced ``C``/``max_iter``
    override them when supplied.
    """

    accepted, ignored = partition(advanced, SKLEARN_ADVANCED_KEYS)
    vectorizer_kwargs: dict[str, Any] = {
        "ngram_range": parse_ngram_range(accepted.get("tfidf_ngram_range", "1,2")),
        "min_df": 1,
    }
    max_features = accepted.get("tfidf_max_features")
    if max_features:
        vectorizer_kwargs["max_features"] = int(max_features)
    class_weight = accepted.get("class_weight")
    classifier_kwargs: dict[str, Any] = {
        "max_iter": int(accepted.get("max_iter") or max(epochs, 1) * 100),
        "C": float(accepted.get("C") or max(learning_rate, 0.0001)),
        "class_weight": "balanced" if class_weight == "balanced" else None,
    }
    return vectorizer_kwargs, classifier_kwargs, ignored


def train_text_classifier(
    dataset_root: Path,
    run_dir: Path,
    labels: list[str],
    epochs: int,
    learning_rate: float,
    advanced: dict | None = None,
) -> tuple[dict, list[dict]]:
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    run_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_kwargs, classifier_kwargs, ignored = tfidf_classifier_settings(
        advanced or {}, epochs, learning_rate
    )
    log_ignored(ignored)
    train = load_text_classification_split(dataset_root, "train")
    valid = load_text_classification_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Text classification training requires labeled train texts")
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    train_texts = [item["text"] for item in train]
    train_y = [int(item["class_id"]) for item in train]
    x_train = vectorizer.fit_transform(train_texts)
    if len(set(train_y)) < 2:
        classifier = DummyClassifier(strategy="constant", constant=train_y[0])
    else:
        classifier = LogisticRegression(**classifier_kwargs)
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
    metrics: dict = average_scores(rows)
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


def tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", text.lower())


def sentences_from_text(text: str) -> list[str]:
    import re

    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if part.strip()]
