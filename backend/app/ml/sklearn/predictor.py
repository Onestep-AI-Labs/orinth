from pathlib import Path

from app.ml.common.base import Predictor
from app.schemas import Detection, InferenceParameters


class SklearnPipelinePredictor(Predictor):
    """Runs an uploaded scikit-learn pipeline over raw text.

    The pipeline is expected to accept a list of raw strings (its own
    vectorizer/transformer stages do feature extraction) and expose
    ``predict`` (and optionally ``predict_proba``) — the same baseline
    predictor contract the built-in text families follow.

    Loading is deferred to first construction (registry lazy dispatch) and the
    heavy joblib/pickle import stays inside this module, off the import path.
    Unpickling executes arbitrary code by design; this is acceptable only under
    the platform's single-user local-trust posture (see docs/ai/rules.md and
    the upload descriptor's security note).
    """

    def __init__(
        self,
        model_path: Path,
        labels: list[str] | None = None,
        *,
        task_type: str = "text_classification",
    ) -> None:
        self.labels = labels or ["positive", "negative", "neutral"]
        self.task_type = task_type
        self.pipeline = self._load(model_path)

    def _load(self, model_path: Path):
        suffix = model_path.suffix.lower()
        if suffix == ".joblib":
            import joblib

            return joblib.load(model_path)
        import pickle

        with model_path.open("rb") as handle:
            return pickle.load(handle)

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        prediction = self.pipeline.predict([text])[0]

        if self.task_type == "summarization":
            return {"task": "summarization", "summary": str(prediction)}
        if self.task_type == "question_answering":
            return {"task": "question_answering", "answer": str(prediction)}

        label = self._resolve_label(prediction)
        scores = self._scores(text, label)
        return {"task": "text_classification", "label": label, "scores": scores}

    def _resolve_label(self, prediction) -> str:
        # Pipelines may emit an integer class index or a string label directly.
        if isinstance(prediction, str):
            return prediction
        try:
            index = int(prediction)
        except (TypeError, ValueError):
            return str(prediction)
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return str(prediction)

    def _scores(self, text: str, label: str) -> dict[str, float]:
        if hasattr(self.pipeline, "predict_proba"):
            try:
                probabilities = self.pipeline.predict_proba([text])[0]
                classes = list(getattr(self.pipeline, "classes_", []))
                scores: dict[str, float] = {}
                for column, class_value in enumerate(classes):
                    key = self._resolve_label(class_value)
                    if column < len(probabilities):
                        scores[key] = round(float(probabilities[column]), 6)
                for known in self.labels:
                    scores.setdefault(known, 0.0)
                return scores
            except Exception:  # noqa: BLE001 — fall back to a one-hot score map
                pass
        return {known: 1.0 if known == label else 0.0 for known in self.labels}
