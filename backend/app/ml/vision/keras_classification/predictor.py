from pathlib import Path

from app.ml.common.base import Predictor
from app.schemas import Detection, InferenceParameters


class KerasClassificationPredictor(Predictor):
    """Runs a saved Keras image classifier on one image.

    The resolution comes from the model itself, not from the registry metadata.
    Those two used to be the same number only by luck: a transfer-learning run
    builds its backbone at the form's `image_size`, so the recorded value
    matched. A model from the architecture studio takes its resolution from the
    graph's Input node and ignores the form entirely, so a graph built at 64×64
    was registered as 512 and every prediction failed with "expected
    shape=(None, 64, 64, 3), found shape=(1, 512, 512, 3)".

    A saved model states what it accepts and physically cannot accept anything
    else, which makes it the only source worth trusting. `image_size` remains as
    a fallback for a model whose input dimensions are undeclared.
    """

    def __init__(self, model_path: Path, labels: list[str], image_size: int = 224) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Keras model not found: {model_path}")
        import tensorflow as tf  # noqa: PLC0415

        from app.ml.architecture.runtime import ensure_registered  # noqa: PLC0415

        # A model from the architecture studio can contain the studio's own
        # layers (SqueezeExcite, PatchEmbedding, RMSNorm, …). They only exist
        # in Keras's registry once their definitions have run, and nothing in
        # this process had run them, so those models failed to load. Harmless
        # for a transfer-learning model, which uses no custom layer.
        ensure_registered()
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.labels = labels
        self.input_size = self._declared_input_size() or (image_size, image_size)

    def _declared_input_size(self) -> tuple[int, int] | None:
        """`(width, height)` the model was built for, or None if it is flexible.

        Kept as a pair rather than one edge because nothing here needs the input
        to be square — only the training pipeline imposes that.
        """

        shape = getattr(self.model, "input_shape", None)
        # A list means multiple inputs, which this predictor does not serve.
        if not isinstance(shape, tuple) or len(shape) != 4:
            return None
        height, width = shape[1], shape[2]
        if not isinstance(height, int) or not isinstance(width, int):
            return None
        return width, height

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def classify(self, image_path: Path, parameters: InferenceParameters) -> dict[str, float]:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        with Image.open(image_path).convert("RGB") as image:
            # PIL takes (width, height); a Keras input shape is (…, height, width, …).
            image = image.resize(self.input_size)
            batch = np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)
        scores = self.model.predict(batch, verbose=0)[0]
        return {
            label: round(float(scores[index]), 6)
            for index, label in enumerate(self.labels)
            if index < len(scores)
        }
