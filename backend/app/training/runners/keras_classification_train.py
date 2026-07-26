import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from app.ml.vision.keras_classification.catalog import KERAS_ADVANCED_PARAMETERS
from app.training.runners.advanced import allowed_keys, log_ignored, parse_advanced, partition

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

    import numpy as np
    import tensorflow as tf
    from PIL import Image

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

    def generator(items):
        for image_path, class_id in items:
            with Image.open(image_path).convert("RGB") as image:
                image = image.resize((args.image_size, args.image_size))
                array = np.asarray(image, dtype=np.float32) / 255.0
            yield array, tf.keras.utils.to_categorical(class_id, num_classes=len(labels))

    output_signature = (
        tf.TensorSpec(shape=(args.image_size, args.image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(labels),), dtype=tf.float32),
    )
    train_ds = (
        tf.data.Dataset.from_generator(lambda: generator(train_items), output_signature=output_signature)
        .shuffle(min(len(train_items), 512))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = (
        tf.data.Dataset.from_generator(lambda: generator(valid_items), output_signature=output_signature)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
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

    csv_path = run_dir / "results.csv"
    callbacks = [
        tf.keras.callbacks.CSVLogger(csv_path),
        tf.keras.callbacks.ModelCheckpoint(
            run_dir / "best_model.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]
    early_stop_patience = int(advanced.get("early_stop_patience", 0) or 0)
    if early_stop_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=early_stop_patience,
                restore_best_weights=True,
            )
        )
    if str(advanced.get("lr_schedule")) == "plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        )
    class_weight = class_weights(train_items, len(labels)) if advanced.get("class_weighting") else None
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    model.save(run_dir / "last_model.keras")
    scores = model.predict(valid_ds, verbose=0)
    y_true = [class_id for _image_path, class_id in valid_items]
    y_pred = [int(np.argmax(row)) for row in scores]
    validation = validation_metrics(y_true, y_pred, scores, labels)
    prediction_rows = [
        {
            "image": str(image_path.name),
            "ground_truth": labels[class_id] if class_id < len(labels) else str(class_id),
            "prediction": labels[y_pred[index]] if y_pred[index] < len(labels) else str(y_pred[index]),
            "scores": {
                label: float(scores[index][label_index])
                for label_index, label in enumerate(labels)
                if label_index < len(scores[index])
            },
        }
        for index, (image_path, class_id) in enumerate(valid_items)
    ]
    metrics: dict[str, Any] = {key: float(values[-1]) for key, values in history.history.items() if values}
    metrics["classes"] = labels
    metrics.update(validation)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "validation_predictions.json").write_text(
        json.dumps(prediction_rows, indent=2),
        encoding="utf-8",
    )
    normalize_results_csv(csv_path)
    print(f"Keras classification training finished. Results saved to: {run_dir}")


def load_labels(dataset_root: Path) -> list[str]:
    manifest = dataset_root / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        labels = [str(label) for label in data.get("labels", []) if str(label).strip()]
        if labels:
            return labels
    return ["class_0", "class_1"]


def load_split(dataset_root: Path, split: str) -> list[tuple[Path, int]]:
    image_dir = dataset_root / split / "images"
    annotation_dir = dataset_root / split / "annotations"
    items: list[tuple[Path, int]] = []
    if not image_dir.exists():
        return items
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        annotation_path = annotation_dir / f"{image_path.stem}.json"
        if not annotation_path.exists():
            continue
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        rows = data.get("annotations", [])
        if not rows:
            continue
        class_id = int(rows[0].get("class_id", 0))
        items.append((image_path, class_id))
    return items


def optimizer_for(name: str, learning_rate, tf):
    normalized = name.lower()
    if normalized == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if normalized == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def learning_rate_schedule(kind: str, base_lr: float, steps_per_epoch: int, epochs: int, tf):
    """Resolve the ``lr_schedule`` advanced knob to a value or Keras schedule.

    ``constant`` and ``plateau`` keep a plain float — plateau adjusts the rate
    through a ``ReduceLROnPlateau`` callback rather than a schedule object.
    """

    total_steps = max(1, steps_per_epoch * max(epochs, 1))
    if kind == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(base_lr, decay_steps=total_steps)
    if kind == "step":
        return tf.keras.optimizers.schedules.ExponentialDecay(
            base_lr,
            decay_steps=max(1, steps_per_epoch * max(epochs // 3, 1)),
            decay_rate=0.1,
            staircase=True,
        )
    return base_lr


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


def class_weights(train_items: list[tuple[Path, int]], num_classes: int) -> dict[int, float]:
    """Balanced per-class weights (``n_samples / (n_classes * count)``)."""

    counts: dict[int, int] = {}
    for _image_path, class_id in train_items:
        counts[class_id] = counts.get(class_id, 0) + 1
    total = sum(counts.values())
    return {
        class_id: total / (num_classes * count)
        for class_id, count in counts.items()
        if count > 0
    }


def validation_metrics(y_true: list[int], y_pred: list[int], scores, labels: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.preprocessing import label_binarize

    label_indices = list(range(len(labels)))
    metrics = {
        "val_accuracy_final": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_indices).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_indices,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "roc_curves": {},
    }
    if not y_true or len(labels) < 2:
        return metrics

    import numpy as np

    y_bin = label_binarize(y_true, classes=label_indices)
    if len(labels) == 2 and y_bin.ndim == 2 and y_bin.shape[1] == 1:
        y_bin = np.concatenate([1 - y_bin, y_bin], axis=1)
    try:
        metrics["macro_auc"] = float(roc_auc_score(y_bin, scores, average="macro", multi_class="ovr"))
        metrics["micro_auc"] = float(roc_auc_score(y_bin, scores, average="micro", multi_class="ovr"))
    except ValueError:
        metrics["macro_auc"] = None
        metrics["micro_auc"] = None

    for index, label in enumerate(labels):
        try:
            fpr, tpr, thresholds = roc_curve(y_bin[:, index], scores[:, index])
        except ValueError:
            continue
        metrics["roc_curves"][label] = {
            "fpr": [float(value) for value in fpr],
            "tpr": [float(value) for value in tpr],
            "thresholds": [float(value) for value in thresholds],
        }
    return metrics


def normalize_results_csv(path: Path) -> None:
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return
    fieldnames = ["epoch", *[name for name in rows[0].keys() if name != "epoch"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            row["epoch"] = index
            writer.writerow(row)


if __name__ == "__main__":
    main()
