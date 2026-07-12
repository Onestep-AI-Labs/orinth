from pathlib import Path

from app.ml.common.base import Predictor
from app.schemas import Detection, InferenceParameters


class KerasClassificationPredictor(Predictor):
    def __init__(self, model_path: Path, labels: list[str], image_size: int = 224) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Keras model not found: {model_path}")
        import tensorflow as tf  # noqa: PLC0415

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.labels = labels
        self.image_size = image_size

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def classify(self, image_path: Path, parameters: InferenceParameters) -> dict[str, float]:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        with Image.open(image_path).convert("RGB") as image:
            image = image.resize((self.image_size, self.image_size))
            batch = np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)
        scores = self.model.predict(batch, verbose=0)[0]
        return {
            label: round(float(scores[index]), 6)
            for index, label in enumerate(self.labels)
            if index < len(scores)
        }
