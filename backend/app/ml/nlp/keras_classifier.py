import json
from pathlib import Path

from app.ml.common.base import Predictor
from app.schemas import Detection, InferenceParameters


class KerasTextClassificationPredictor(Predictor):
    """Runs a saved Keras text classifier on one string.

    The sequence length is read off the model, with the recorded `max_length` as
    a fallback. Same reasoning as the image classifier: an architecture-studio
    model takes its length from the graph's Input node rather than from the
    training form, so metadata and model can disagree — and only one of them is
    what the weights were actually built for.
    """

    def __init__(self, model_path: Path, labels: list[str] | None = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Keras NLP model not found: {model_path}")
        import tensorflow as tf  # noqa: PLC0415

        self.model_path = model_path
        self.model_dir = model_path.parent
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.metadata = _read_json(self.model_dir / "metadata.json")
        self.labels = labels or self.metadata.get("labels") or ["positive", "negative", "neutral"]
        self.max_length = self._declared_sequence_length() or int(
            self.metadata.get("max_length") or 160
        )
        tokenizer_json = (self.model_dir / "tokenizer.json").read_text(encoding="utf-8")
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        self.pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

    def _declared_sequence_length(self) -> int | None:
        """Tokens per example the model was built for, or None if it is flexible."""

        shape = getattr(self.model, "input_shape", None)
        if not isinstance(shape, tuple) or len(shape) != 2:
            return None
        return shape[1] if isinstance(shape[1], int) else None

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = self.pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")
        scores_raw = self.model.predict(padded, verbose=0)[0]
        scores = {
            label: round(float(scores_raw[index]), 6)
            for index, label in enumerate(self.labels)
            if index < len(scores_raw)
        }
        scores.update({label: scores.get(label, 0.0) for label in self.labels})
        label = max(scores.items(), key=lambda item: item[1])[0] if scores else self.labels[0]
        return {"task": "text_classification", "label": label, "scores": scores}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
