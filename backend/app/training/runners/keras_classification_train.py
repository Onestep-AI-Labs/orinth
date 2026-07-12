import argparse
import csv
import json
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf
    from PIL import Image

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
    base.trainable = False
    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    optimizer = optimizer_for(args.optimizer, args.learning_rate, tf)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

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
    history = model.fit(train_ds, validation_data=valid_ds, epochs=args.epochs, callbacks=callbacks)
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


def optimizer_for(name: str, learning_rate: float, tf):
    normalized = name.lower()
    if normalized == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if normalized == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


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
