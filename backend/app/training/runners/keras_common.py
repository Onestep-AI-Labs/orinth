"""Dataset loading, optimizers, and metrics shared by the Keras runners.

Extracted from `keras_classification_train.py` when the phase 17 architecture
runner needed the same image-classification plumbing. Behavior is unchanged —
both runners now read the same dataset layout, write the same `metrics.json`
and `results.csv`, and therefore promote into the model registry through the
same code path.

TensorFlow, numpy, PIL, and sklearn are imported inside functions: these
helpers are imported by runner modules at argument-parsing time, before the
heavy libraries should be paid for.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# How many decoded images the tf.data shuffle holds. It cannot be the dataset
# size: the buffer holds fully decoded tensors, and 1770 images at 224² would be
# a gigabyte. Mixing is guaranteed by permuting the file list first (see
# `image_datasets`); this buffer only varies the order between epochs.
SHUFFLE_BUFFER = 512


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
    for image_path in sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    ):
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


def image_datasets(
    train_items: list[tuple[Path, int]],
    valid_items: list[tuple[Path, int]],
    *,
    image_size: int,
    batch_size: int,
    num_labels: int,
    tf,
):
    """`(train_ds, valid_ds)` of batched, prefetched `(image, one-hot)` pairs.

    The training file list is permuted here, before tf.data sees it, and that is
    not a nicety.

    `load_split` walks the split directory in sorted filename order, and every
    dataset the platform writes names files by class — so the list arrives in
    perfect class order: 282 cardboard, then 351 glass, then 287 metal, and so
    on. A tf.data `shuffle` only mixes within its buffer, and the buffer holds
    decoded images so it cannot be the size of the dataset. With a 512-image
    buffer over a class-ordered list of 1770, every batch in an epoch contained
    at most two classes. The model chased whichever class was streaming past,
    training loss *climbed* through the epoch as the distribution moved under
    it, and validation accuracy sat at chance.

    Permuting the list costs nothing — it is a list of paths — and it is exact
    rather than approximate. The buffered shuffle stays on top of it so the
    order still differs between epochs. Determinism is preserved: the runners
    call `keras.utils.set_random_seed`, which seeds Python's `random`.
    """

    import numpy as np
    from PIL import Image

    train_items = list(train_items)
    random.shuffle(train_items)

    def generator(items):
        for image_path, class_id in items:
            with Image.open(image_path).convert("RGB") as image:
                image = image.resize((image_size, image_size))
                array = np.asarray(image, dtype=np.float32) / 255.0
            yield array, tf.keras.utils.to_categorical(class_id, num_classes=num_labels)

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(num_labels,), dtype=tf.float32),
    )

    def batches(items: list[tuple[Path, int]]) -> int:
        return max(1, -(-len(items) // max(batch_size, 1)))

    def pipeline(items: list[tuple[Path, int]], *, shuffle: bool):
        dataset = tf.data.Dataset.from_generator(
            lambda: generator(items), output_signature=output_signature
        )
        if shuffle:
            dataset = dataset.shuffle(min(len(items), SHUFFLE_BUFFER))
        # A generator dataset has unknown cardinality, so Keras counts steps as
        # it goes: the first epoch renders as "1/Unknown" and every run ends on
        # a spurious "your input ran out of data" warning. The item count is
        # known here, so state it and the progress bar reads normally.
        return (
            dataset.batch(batch_size)
            .apply(tf.data.experimental.assert_cardinality(batches(items)))
            .prefetch(tf.data.AUTOTUNE)
        )

    return pipeline(train_items, shuffle=True), pipeline(valid_items, shuffle=False)


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


def standard_callbacks(run_dir: Path, advanced: dict, tf) -> list:
    """CSV logging, best-checkpointing, and the optional early-stop / plateau pair."""

    callbacks = [
        tf.keras.callbacks.CSVLogger(run_dir / "results.csv"),
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
    return callbacks


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


def write_evaluation(
    run_dir: Path,
    model,
    valid_ds,
    valid_items: list[tuple[Path, int]],
    labels: list[str],
    history_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Score the validation split and write `metrics.json` + predictions.

    Both Keras runners write the identical artifact set, which is what lets
    `_register_training_model` promote either one through the same branch.
    """

    import numpy as np

    scores = model.predict(valid_ds, verbose=0)
    y_true = [class_id for _image_path, class_id in valid_items]
    y_pred = [int(np.argmax(row)) for row in scores]
    metrics: dict[str, Any] = dict(history_metrics)
    metrics["classes"] = labels
    metrics.update(validation_metrics(y_true, y_pred, scores, labels))

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
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "validation_predictions.json").write_text(
        json.dumps(prediction_rows, indent=2), encoding="utf-8"
    )
    normalize_results_csv(run_dir / "results.csv")
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
