"""The shared Keras image pipeline.

The regression these cover: `load_split` returns items in sorted filename order,
and every dataset the platform writes names files by class, so the list arrives
in perfect class order. A tf.data `shuffle` only mixes within its buffer, and
the buffer holds decoded images so it cannot be dataset-sized — which meant
every batch of a run contained at most one or two classes, training loss climbed
through each epoch, and validation accuracy stayed at chance.
"""

import json
from pathlib import Path

import pytest

from app.training.runners.keras_common import SHUFFLE_BUFFER, image_datasets, load_split


def write_split(root: Path, split: str, counts: dict[int, int]) -> None:
    """A split whose filenames sort into contiguous per-class runs, as ours do."""

    from PIL import Image

    images = root / split / "images"
    annotations = root / split / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for class_id, count in sorted(counts.items()):
        for index in range(count):
            stem = f"class{class_id}_{index:04d}"
            Image.new("RGB", (8, 8)).save(images / f"{stem}.png")
            (annotations / f"{stem}.json").write_text(
                json.dumps({"annotations": [{"class_id": class_id}]}), encoding="utf-8"
            )


def test_load_split_returns_items_in_class_order(tmp_path: Path):
    """Documents the input condition the shuffle has to survive."""

    write_split(tmp_path, "train", {0: 5, 1: 5, 2: 5})

    ids = [class_id for _path, class_id in load_split(tmp_path, "train")]

    assert ids == [0] * 5 + [1] * 5 + [2] * 5


@pytest.mark.slow
def test_batches_mix_every_class_even_when_the_split_is_class_ordered(tmp_path: Path):
    """The real assertion: a batch must be a sample of the whole label set.

    Six classes in contiguous runs over a split larger than `SHUFFLE_BUFFER` —
    the shape of the trash dataset that exposed this. A sliding buffer does mix
    somewhat, so the failure is statistical rather than absolute: measured on
    this fixture, buffered shuffling alone puts all six classes in 19 of 38
    batches, and permuting the list first puts them in all 38. The batches it
    misses are the ones near a class boundary, where the composition tracks
    position in the stream instead of the dataset — which is what makes the
    loss climb through an epoch.
    """

    import numpy as np
    import tensorflow as tf

    classes = 6
    write_split(tmp_path, "train", {class_id: 200 for class_id in range(classes)})
    items = load_split(tmp_path, "train")
    assert len(items) > SHUFFLE_BUFFER

    train_ds, _valid_ds = image_datasets(
        items, items, image_size=8, batch_size=32, num_labels=classes, tf=tf
    )

    seen = [set(np.argmax(labels.numpy(), axis=1).tolist()) for _images, labels in train_ds]
    complete = sum(1 for batch in seen if len(batch) == classes)
    assert complete > len(seen) * 0.8, (
        f"only {complete} of {len(seen)} batches saw all {classes} classes"
    )


@pytest.mark.slow
def test_the_pipeline_declares_its_length(tmp_path: Path):
    """Known cardinality is what keeps Keras from printing `1/Unknown` and then
    warning that the input ran out of data."""

    import tensorflow as tf

    write_split(tmp_path, "train", {0: 5, 1: 5})
    write_split(tmp_path, "valid", {0: 2, 1: 3})
    train_items = load_split(tmp_path, "train")
    valid_items = load_split(tmp_path, "valid")

    train_ds, valid_ds = image_datasets(
        train_items, valid_items, image_size=8, batch_size=4, num_labels=2, tf=tf
    )

    assert int(train_ds.cardinality()) == 3  # ceil(10 / 4)
    assert int(valid_ds.cardinality()) == 2  # ceil(5 / 4)
    # A stated cardinality that disagrees with the stream raises on iteration,
    # so counting the batches also proves the claim is true.
    assert sum(1 for _batch in train_ds) == 3
    assert sum(1 for _batch in valid_ds) == 2


@pytest.mark.slow
def test_validation_order_is_left_alone(tmp_path: Path):
    """`write_evaluation` zips predictions against `valid_items` positionally,
    so shuffling the validation split would mislabel every prediction."""

    import numpy as np
    import tensorflow as tf

    write_split(tmp_path, "valid", {0: 4, 1: 4})
    items = load_split(tmp_path, "valid")

    _train_ds, valid_ds = image_datasets(
        items, items, image_size=8, batch_size=8, num_labels=2, tf=tf
    )

    order = [
        int(value)
        for _images, labels in valid_ds
        for value in np.argmax(labels.numpy(), axis=1)
    ]
    assert order == [class_id for _path, class_id in items]
