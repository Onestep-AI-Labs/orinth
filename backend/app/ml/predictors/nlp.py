import pickle
import re
from pathlib import Path

from app.ml.predictors.base import Predictor
from app.schemas import Detection, InferenceParameters


class TextClassificationPredictor(Predictor):
    def __init__(self, model_path: Path | None = None, labels: list[str] | None = None) -> None:
        self.labels = labels or ["positive", "negative", "neutral"]
        self.vectorizer = None
        self.classifier = None
        if model_path and model_path.exists():
            with model_path.open("rb") as handle:
                payload = pickle.load(handle)
            self.vectorizer = payload.get("vectorizer")
            self.classifier = payload.get("classifier")
            self.labels = payload.get("labels") or self.labels

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        if self.vectorizer is not None and self.classifier is not None:
            matrix = self.vectorizer.transform([text])
            pred_index = int(self.classifier.predict(matrix)[0])
            scores = {}
            if hasattr(self.classifier, "predict_proba"):
                probabilities = self.classifier.predict_proba(matrix)[0]
                class_values = [int(value) for value in getattr(self.classifier, "classes_", [])]
                scores = {
                    self.labels[class_id]: round(float(probabilities[column_index]), 6)
                    for column_index, class_id in enumerate(class_values)
                    if 0 <= class_id < len(self.labels) and column_index < len(probabilities)
                }
                scores.update({label: scores.get(label, 0.0) for label in self.labels})
            else:
                scores = {label: 1.0 if index == pred_index else 0.0 for index, label in enumerate(self.labels)}
            label = self.labels[pred_index] if 0 <= pred_index < len(self.labels) else str(pred_index)
            return {"task": "text_classification", "label": label, "scores": scores}

        scores = self._keyword_scores(text)
        label = max(scores.items(), key=lambda item: item[1])[0]
        return {"task": "text_classification", "label": label, "scores": scores}

    def _keyword_scores(self, text: str) -> dict[str, float]:
        tokens = set(_tokens(text))
        positive = {"good", "great", "excellent", "clear", "helpful", "fast", "stable", "success"}
        negative = {"bad", "poor", "slow", "broken", "error", "failed", "risk", "issue"}
        raw = {
            "positive": 1 + len(tokens & positive),
            "negative": 1 + len(tokens & negative),
            "neutral": 1,
        }
        total = sum(raw.get(label, 1) for label in self.labels) or 1
        return {label: round(raw.get(label, 1) / total, 6) for label in self.labels}


class ExtractiveSummarizerPredictor(Predictor):
    def __init__(self, model_path: Path | None = None, labels: list[str] | None = None) -> None:
        self.labels = labels or ["summary"]
        self.keywords: list[str] = []
        if model_path and model_path.exists():
            payload = _read_json_like(model_path)
            self.keywords = [str(item) for item in payload.get("keywords", [])]

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        sentences = _sentences(text)
        if not sentences:
            return {"task": "summarization", "summary": ""}
        keywords = set(self.keywords or _tokens(text))
        ranked = sorted(
            sentences,
            key=lambda sentence: (len(set(_tokens(sentence)) & keywords), -len(sentence)),
            reverse=True,
        )
        summary = ranked[0]
        max_chars = max(parameters.max_length, 1)
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] or summary[:max_chars]
        return {"task": "summarization", "summary": summary}


class KeywordQAPredictor(Predictor):
    def __init__(self, model_path: Path | None = None, labels: list[str] | None = None) -> None:
        self.labels = labels or ["answer"]
        self.examples: list[dict] = []
        if model_path and model_path.exists():
            payload = _read_json_like(model_path)
            self.examples = [item for item in payload.get("examples", []) if isinstance(item, dict)]

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        question = parameters.question or ""
        question_tokens = set(_tokens(question))
        sentences = _sentences(text)
        if not sentences:
            return {"task": "question_answering", "answer": "", "score": 0.0}
        best = max(
            sentences,
            key=lambda sentence: len(set(_tokens(sentence)) & question_tokens),
        )
        overlap = len(set(_tokens(best)) & question_tokens)
        score = overlap / max(len(question_tokens), 1)
        return {"task": "question_answering", "answer": best, "score": round(float(score), 6)}


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [part.strip() for part in parts if part.strip()]


def _read_json_like(path: Path) -> dict:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
