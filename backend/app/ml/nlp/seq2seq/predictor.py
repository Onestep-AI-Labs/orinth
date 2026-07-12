import json
from pathlib import Path

from app.ml.common.base import Predictor
from app.schemas import Detection, InferenceParameters


class KerasSeq2SeqSummarizerPredictor(Predictor):
    def __init__(self, model_path: Path, labels: list[str] | None = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Keras seq2seq model not found: {model_path}")
        import tensorflow as tf  # noqa: PLC0415

        self.model_path = model_path
        self.model_dir = model_path.parent
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.metadata = _read_json(self.model_dir / "metadata.json")
        tokenizer_json = (self.model_dir / "tokenizer.json").read_text(encoding="utf-8")
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        self.pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
        self.input_max_length = int(self.metadata.get("max_length") or 192)
        self.target_max_length = int(self.metadata.get("target_max_length") or 48)
        self.start_token = str(self.metadata.get("start_token") or "startseq")
        self.end_token = str(self.metadata.get("end_token") or "endseq")
        self.index_word = {int(key): value for key, value in self.tokenizer.index_word.items()}
        self.start_id = int(self.tokenizer.word_index.get(self.start_token, 0))
        self.end_id = int(self.tokenizer.word_index.get(self.end_token, 0))

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        source = self.tokenizer.texts_to_sequences([text])
        encoder_input = self.pad_sequences(
            source,
            maxlen=self.input_max_length,
            padding="post",
            truncating="post",
        )
        decoder = [self.start_id]
        max_tokens = min(max(parameters.max_length, 1), self.target_max_length)
        for _ in range(max_tokens):
            decoder_input = self.pad_sequences(
                [decoder],
                maxlen=self.target_max_length,
                padding="post",
                truncating="post",
            )
            prediction = self.model.predict([encoder_input, decoder_input], verbose=0)[0]
            next_position = min(len(decoder) - 1, prediction.shape[0] - 1)
            next_id = int(prediction[next_position].argmax())
            if next_id in {0, self.end_id}:
                break
            decoder.append(next_id)
        words = [
            self.index_word.get(token_id, "")
            for token_id in decoder[1:]
            if token_id not in {0, self.start_id, self.end_id}
        ]
        summary = " ".join(word for word in words if word).strip()
        return {"task": "summarization", "summary": summary}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
