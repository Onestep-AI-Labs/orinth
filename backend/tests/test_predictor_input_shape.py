"""Predictors take their input geometry from the model, not from metadata.

The regression: a transfer-learning run builds its backbone at the training
form's `image_size`, so the recorded value and the model agreed by construction.
An architecture-studio model takes its resolution from the graph's Input node and
ignores the form, so a 64x64 graph was registered as 512 and every prediction
failed with "expected shape=(None, 64, 64, 3), found shape=(1, 512, 512, 3)".
Testing and inference both went through this, so both were dead.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def keras_image_model(tmp_path: Path) -> Path:
    import tensorflow as tf

    inputs = tf.keras.Input((64, 64, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    path = tmp_path / "best_model.keras"
    tf.keras.Model(inputs, outputs).save(path)
    return path


@pytest.mark.slow
def test_image_predictor_ignores_a_metadata_size_the_model_contradicts(
    keras_image_model: Path, tmp_path: Path
):
    from PIL import Image

    from app.ml.vision.keras_classification.predictor import KerasClassificationPredictor
    from app.schemas import InferenceParameters

    # What the registry recorded for these runs: the training form's default.
    predictor = KerasClassificationPredictor(
        keras_image_model, ["a", "b", "c"], image_size=512
    )

    assert predictor.input_size == (64, 64)

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (800, 600), "white").save(image_path)
    scores = predictor.classify(image_path, InferenceParameters())

    assert set(scores) == {"a", "b", "c"}
    assert abs(sum(scores.values()) - 1.0) < 1e-4


@pytest.mark.slow
def test_image_predictor_falls_back_when_the_model_declares_no_size(tmp_path: Path):
    """A fully convolutional model accepts any resolution, so metadata decides."""

    import tensorflow as tf
    from PIL import Image

    from app.ml.vision.keras_classification.predictor import KerasClassificationPredictor
    from app.schemas import InferenceParameters

    inputs = tf.keras.Input((None, None, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    path = tmp_path / "flexible.keras"
    tf.keras.Model(inputs, outputs).save(path)

    predictor = KerasClassificationPredictor(path, ["a", "b"], image_size=128)

    assert predictor.input_size == (128, 128)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (300, 200), "white").save(image_path)
    assert set(predictor.classify(image_path, InferenceParameters())) == {"a", "b"}


@pytest.mark.slow
def test_text_predictor_takes_its_sequence_length_from_the_model(tmp_path: Path):
    """The same rule for text graphs, whose length comes from the Input node."""

    import tensorflow as tf

    from app.ml.nlp.keras_classifier import KerasTextClassificationPredictor
    from app.schemas import InferenceParameters

    inputs = tf.keras.Input((32,), dtype="int32")
    x = tf.keras.layers.Embedding(50, 8)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model_path = tmp_path / "best_model.keras"
    tf.keras.Model(inputs, outputs).save(model_path)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50, oov_token="<OOV>")
    tokenizer.fit_on_texts(["spam offer now", "meeting notes monday"])
    (tmp_path / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    # Deliberately wrong: the training form's default, not the graph's length.
    (tmp_path / "metadata.json").write_text(
        json.dumps({"labels": ["spam", "ham"], "max_length": 160}), encoding="utf-8"
    )

    predictor = KerasTextClassificationPredictor(model_path, None)

    assert predictor.max_length == 32
    result = predictor.predict_text("spam offer now", InferenceParameters())
    assert result["task"] == "text_classification"
    assert result["label"] in {"spam", "ham"}
