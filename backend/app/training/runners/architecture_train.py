"""Train a model built by the architecture studio (phase 17).

The service emits `generated_model.py` into the run directory from the saved
graph; this runner imports it and calls `build_model(num_classes=...)`. There
is no second definition of what a graph means — the file that trains is the
same file the user downloads from the Code panel.

Everything after the model is built is `keras_common`, so a graph-built model
produces byte-for-byte the same artifact set as the stock Keras classification
runner and promotes into the registry through the same branch.
"""

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Any

from app.ml.architecture.train_catalog import ARCHITECTURE_ADVANCED_PARAMETERS
from app.training.runners.advanced import allowed_keys, log_ignored, parse_advanced, partition
from app.training.runners.keras_common import (
    class_weights,
    image_datasets,
    learning_rate_schedule,
    load_labels,
    load_split,
    optimizer_for,
    standard_callbacks,
    write_evaluation,
)

ARCHITECTURE_ADVANCED_KEYS = allowed_keys(ARCHITECTURE_ADVANCED_PARAMETERS)


def select_advanced(raw: str | None) -> tuple[dict[str, Any], list[str]]:
    return partition(parse_advanced(raw), ARCHITECTURE_ADVANCED_KEYS)


def load_build_model(module_path: Path):
    """Import the generated module and return its `build_model`.

    This is where user-authored graph code is executed, in the training
    subprocess — never in the API process. See the phase 17 spec's
    "Custom code and trust" section.
    """

    spec = importlib.util.spec_from_file_location("generated_model", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load the generated model at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model"):
        raise ValueError("The generated model module has no build_model() function")
    return module.build_model


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
        raise ValueError("Training requires annotated train images")
    if not valid_items:
        valid_items = train_items

    build_model = load_build_model(Path(args.model_file))
    model = build_model(num_classes=len(labels))
    model.summary()

    # The Input node owns the resolution, so preprocessing reads it off the
    # built model rather than taking a separate image-size argument that could
    # silently disagree with the graph.
    image_size = input_edge(model)
    if model.output_shape[-1] != len(labels):
        raise ValueError(
            f"This architecture outputs {model.output_shape[-1]} units but the dataset has "
            f"{len(labels)} classes. Set the output Dense layer to "
            '"Units = dataset class count", or pick a matching dataset.'
        )

    train_ds, valid_ds = image_datasets(
        train_items,
        valid_items,
        image_size=image_size,
        batch_size=args.batch_size,
        num_labels=len(labels),
        tf=tf,
    )

    steps_per_epoch = max(1, math.ceil(len(train_items) / max(args.batch_size, 1)))
    learning_rate = learning_rate_schedule(
        str(advanced.get("lr_schedule", "constant")),
        args.learning_rate,
        steps_per_epoch,
        args.epochs,
        tf,
    )
    model.compile(
        optimizer=optimizer_for(args.optimizer, learning_rate, tf),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=float(advanced.get("label_smoothing", 0.0))
        ),
        metrics=["accuracy"],
    )

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
    print(f"Architecture training finished. Results saved to: {run_dir}")


def input_edge(model) -> int:
    """Square input edge length from the built model's input shape."""

    shape = model.input_shape
    if isinstance(shape, list):
        raise ValueError("Multi-input architectures are not trainable from the image pipeline yet")
    if len(shape) != 4:
        raise ValueError(
            f"Image training needs a rank-3 input (height, width, channels); this graph has {shape[1:]}"
        )
    height, width = shape[1], shape[2]
    if height is None or width is None:
        raise ValueError("The Input node must declare a fixed height and width for image training")
    if height != width:
        raise ValueError(
            f"Non-square inputs are not supported yet; this graph asks for {height}x{width}"
        )
    return int(height)


if __name__ == "__main__":
    main()
