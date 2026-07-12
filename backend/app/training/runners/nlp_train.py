import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    labels = load_labels(dataset_root)

    if args.task_type == "text_classification":
        metrics, predictions = train_text_classifier(
            dataset_root, run_dir, labels, max(args.epochs, 1), args.learning_rate
        )
    elif args.task_type == "summarization":
        metrics, predictions = train_summarizer(dataset_root, run_dir, max(args.epochs, 1))
    elif args.task_type == "question_answering":
        metrics, predictions = train_qa(dataset_root, run_dir, max(args.epochs, 1))
    else:
        raise ValueError(f"Unsupported NLP task type: {args.task_type}")

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


if __name__ == "__main__":
    main()
