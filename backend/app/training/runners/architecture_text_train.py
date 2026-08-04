"""Train a studio-built graph on labelled text (phase 17, stage F).

The sibling of `architecture_train`: same generated module, same
`build_model(num_classes=...)` contract, a different dataset pipeline in front
of it. A text graph starts at an Input node carrying a sequence length and an
Embedding node carrying a vocabulary size, so the tokenizer is configured from
the graph rather than from a form — the same rule the image runner follows when
it reads its resolution off the built model.

Artifacts match `nlp/keras.py`'s text classifier byte for byte in shape, so a
graph-built classifier promotes, tests, and serves through the existing
`KerasTextClassificationPredictor` with no knowledge that a canvas produced it.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from app.ml.architecture.train_catalog import ARCHITECTURE_ADVANCED_PARAMETERS
from app.training.runners.advanced import allowed_keys, log_ignored, parse_advanced, partition
from app.training.runners.architecture_train import emits_probabilities, load_build_model
from app.training.runners.keras_common import (
    class_weights,
    learning_rate_schedule,
    optimizer_for,
    standard_callbacks,
)
from app.training.runners.nlp.common import load_labels, load_text_classification_split

ARCHITECTURE_ADVANCED_KEYS = allowed_keys(ARCHITECTURE_ADVANCED_PARAMETERS)


def select_advanced(raw: str | None) -> tuple[dict[str, Any], list[str]]:
    return partition(parse_advanced(raw), ARCHITECTURE_ADVANCED_KEYS)


def sequence_length(model) -> int:
    """Token count per example, from the graph's Input node."""

    shape = model.input_shape
    if isinstance(shape, list):
        raise ValueError("Multi-input architectures are not trainable from the text pipeline yet")
    if len(shape) != 2:
        raise ValueError(
            "Text training needs a rank-1 input (sequence length); this graph's Input node "
            f"declares {shape[1:]}. Set it to a single number, for example 128."
        )
    if shape[1] is None:
        raise ValueError("The Input node must declare a fixed sequence length for text training")
    return int(shape[1])


def embedding_vocabulary(model) -> int | None:
    """The first Embedding layer's table size, or None if the graph has none.

    The tokenizer is capped to this so no example can index past the end of the
    embedding matrix — the one way a text graph fails at the first batch rather
    than at build time.
    """

    for layer in model.layers:
        input_dim = getattr(layer, "input_dim", None)
        if input_dim is not None and hasattr(layer, "output_dim"):
            return int(input_dim)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--advanced", default="{}")
    args = parser.parse_args()

    advanced, ignored = select_advanced(args.advanced)
    log_ignored(ignored)

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    if "seed" in advanced:
        tf.keras.utils.set_random_seed(int(advanced["seed"]))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    labels = load_labels(dataset_root)
    train = load_text_classification_split(dataset_root, "train")
    valid = load_text_classification_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Training requires annotated train texts")

    build_model = load_build_model(Path(args.model_file))
    model = build_model(num_classes=len(labels))
    model.summary()

    if model.output_shape[-1] != len(labels):
        raise ValueError(
            f"This architecture outputs {model.output_shape[-1]} units but the dataset has "
            f"{len(labels)} classes. Set the output Dense layer to "
            '"Units = dataset class count", or pick a matching dataset.'
        )

    max_length = sequence_length(model)
    vocab_size = embedding_vocabulary(model)
    if vocab_size is None:
        raise ValueError(
            "This graph has no Embedding node, so token ids have nothing to look up. "
            "Add one between the Input node and the rest of the model."
        )

    train_texts = [item["text"] for item in train]
    valid_texts = [item["text"] for item in valid]
    train_y = np.asarray([int(item["class_id"]) for item in train], dtype="int32")
    valid_y = np.asarray([int(item["class_id"]) for item in valid], dtype="int32")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    print(
        f"Sequence length {max_length} and vocabulary {vocab_size} taken from the graph; "
        f"{len(tokenizer.word_index)} distinct tokens seen in training text"
    )
    x_train = pad(tf, tokenizer, train_texts, max_length)
    x_valid = pad(tf, tokenizer, valid_texts, max_length)

    steps_per_epoch = max(1, -(-len(train) // max(args.batch_size, 1)))
    learning_rate = learning_rate_schedule(
        str(advanced.get("lr_schedule", "constant")),
        args.learning_rate,
        steps_per_epoch,
        args.epochs,
        tf,
    )
    # Same reasoning as the image runner: a head left on the catalog's default
    # `linear` activation emits logits, and probability cross-entropy over those
    # trains at chance without ever raising.
    from_logits = not emits_probabilities(model)
    if from_logits:
        print(
            "Output layer has no softmax activation; compiling the loss with "
            "from_logits=True. Set the head's activation to softmax on the canvas "
            "to make the graph self-describing."
        )
    model.compile(
        optimizer=optimizer_for(args.optimizer, learning_rate, tf),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=from_logits,
            label_smoothing=float(advanced.get("label_smoothing", 0.0)),
        ),
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        tf.keras.utils.to_categorical(train_y, num_classes=len(labels)),
        validation_data=(
            x_valid,
            tf.keras.utils.to_categorical(valid_y, num_classes=len(labels)),
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=standard_callbacks(run_dir, advanced, tf),
        class_weight=class_weights(
            [(Path(item["id"]), int(item["class_id"])) for item in train], len(labels)
        )
        if advanced.get("class_weighting")
        else None,
        verbose=2,
    )
    model.save(run_dir / "last_model.keras")
    if not (run_dir / "best_model.keras").exists():
        model.save(run_dir / "best_model.keras")
    (run_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")

    probabilities = model.predict(x_valid, verbose=0)
    y_pred = [int(value) for value in np.argmax(probabilities, axis=1)]
    y_true = [int(value) for value in valid_y.tolist()]
    class_indices = list(range(len(labels)))
    metrics = {
        "classes": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_indices).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=class_indices,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        **{key: float(values[-1]) for key, values in history.history.items() if values},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # `max_length` and the label set are not recoverable from the weights, and
    # the predictor needs both to tokenize an incoming string the same way.
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "keras_text_classifier",
                "model_kind": "architecture_graph",
                "labels": labels,
                "max_length": max_length,
                "vocab_size": vocab_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "validation_predictions.json").write_text(
        json.dumps(
            [
                {
                    "item": item["id"],
                    "ground_truth": labels[y_true[index]],
                    "prediction": labels[y_pred[index]],
                    "scores": {
                        labels[class_index]: float(probabilities[index][class_index])
                        for class_index in class_indices
                    },
                }
                for index, item in enumerate(valid)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    # `results.csv` is already written by the CSVLogger in `standard_callbacks`.
    print(f"Architecture text training finished. Results saved to: {run_dir}")


def pad(tf, tokenizer, texts: list[str], max_length: int):
    sequences = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding="post", truncating="post"
    )


if __name__ == "__main__":
    main()
