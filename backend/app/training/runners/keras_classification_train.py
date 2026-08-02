import argparse
import json
import math
from pathlib import Path
from typing import Any

from app.ml.vision.keras_classification.catalog import KERAS_ADVANCED_PARAMETERS
from app.training.runners.advanced import allowed_keys, log_ignored, parse_advanced, partition
from app.training.runners.keras_common import (
    IMAGE_SUFFIXES,  # noqa: F401 - re-exported for callers that imported it from here
    class_weights,
    image_datasets,
    learning_rate_schedule,
    load_labels,
    load_split,
    normalize_results_csv,  # noqa: F401 - re-exported; tests import it from this module
    optimizer_for,
    standard_callbacks,
    validation_metrics,  # noqa: F401 - re-exported
    write_evaluation,
)

KERAS_ADVANCED_KEYS = allowed_keys(KERAS_ADVANCED_PARAMETERS)


def select_keras_advanced(raw: str | None) -> tuple[dict[str, Any], list[str]]:
    """Return ``(accepted advanced values, ignored keys)`` for the Keras runner."""

    return partition(parse_advanced(raw), KERAS_ADVANCED_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-model", default="EfficientNetB0")
    parser.add_argument("--application-kwargs", default="{}")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--advanced", default="{}")
    args = parser.parse_args()

    advanced, ignored = select_keras_advanced(args.advanced)
    log_ignored(ignored)

    import tensorflow as tf

    if "seed" in advanced:
        tf.keras.utils.set_random_seed(int(advanced["seed"]))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    labels = load_labels(dataset_root)
    train_items = load_split(dataset_root, "train")
    valid_items = load_split(dataset_root, "valid")
    if not train_items:
        raise ValueError("Classification training requires annotated train images")
    if not valid_items:
        valid_items = train_items

    train_ds, valid_ds = image_datasets(
        train_items,
        valid_items,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_labels=len(labels),
        tf=tf,
    )

    app = getattr(tf.keras.applications, args.base_model)
    application_kwargs = json.loads(args.application_kwargs)
    base = app(
        weights="imagenet",
        include_top=False,
        input_shape=(args.image_size, args.image_size, 3),
        pooling="avg",
        **application_kwargs,
    )
    # Fine-tune the last N backbone layers when asked; otherwise the backbone
    # stays frozen and only the fresh head trains.
    unfreeze_layers = int(advanced.get("unfreeze_layers", 0) or 0)
    if unfreeze_layers > 0:
        base.trainable = True
        for layer in base.layers[:-unfreeze_layers]:
            layer.trainable = False
    else:
        base.trainable = False
    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
    augmentation = build_augmentation(advanced, tf)
    x = augmentation(inputs) if augmentation is not None else inputs
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(float(advanced.get("dropout", 0.2)))(x)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    steps_per_epoch = max(1, math.ceil(len(train_items) / max(args.batch_size, 1)))
    learning_rate = learning_rate_schedule(
        str(advanced.get("lr_schedule", "constant")),
        args.learning_rate,
        steps_per_epoch,
        args.epochs,
        tf,
    )
    optimizer = optimizer_for(args.optimizer, learning_rate, tf)
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=float(advanced.get("label_smoothing", 0.0))
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=standard_callbacks(run_dir, advanced, tf),
        class_weight=class_weights(train_items, len(labels))
        if advanced.get("class_weighting")
        else None,
    )
    model.save(run_dir / "last_model.keras")
    write_evaluation(
        run_dir,
        model,
        valid_ds,
        valid_items,
        labels,
        {key: float(values[-1]) for key, values in history.history.items() if values},
    )
    print(f"Keras classification training finished. Results saved to: {run_dir}")


def build_augmentation(advanced: dict, tf):
    """A Keras augmentation stack for the enabled toggles, or ``None``."""

    layers = []
    if advanced.get("aug_horizontal_flip"):
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if advanced.get("aug_rotation"):
        layers.append(tf.keras.layers.RandomRotation(0.1))
    if advanced.get("aug_zoom"):
        layers.append(tf.keras.layers.RandomZoom(0.1))
    if advanced.get("aug_contrast"):
        layers.append(tf.keras.layers.RandomContrast(0.1))
    if not layers:
        return None
    return tf.keras.Sequential(layers, name="augmentation")


if __name__ == "__main__":
    main()
