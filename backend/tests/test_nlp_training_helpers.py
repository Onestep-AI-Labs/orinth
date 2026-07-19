"""Unit coverage for the Hugging Face training loop's non-obvious behaviour.

These are the two defects that made a fine-tune look fine while being wrong:
validation silently falling back to the training set, and unshuffled batches
that group by label. Both are cheap to test without loading a model.
"""

import random

from app.training.runners.nlp.huggingface import _resolve_validation, _shuffled_batches


def _items(count: int, class_ids: list[int] | None = None) -> list[dict]:
    return [
        {"id": f"item-{index}", "text": f"text {index}", "class_id": (class_ids or [0] * count)[index]}
        for index in range(count)
    ]


def test_uses_the_valid_split_when_it_is_large_enough():
    train = _items(20)
    valid = _items(8)

    resolved_train, resolved_valid, source = _resolve_validation(train, valid, seed=42)

    assert source == "valid_split"
    assert resolved_train == train
    assert resolved_valid == valid


def test_carves_a_holdout_rather_than_scoring_on_the_training_set():
    """An empty valid split must not silently become the training set.

    `valid = load(...) or train` reported training accuracy as validation
    accuracy, so the model scored best exactly when it had memorised.
    """
    train = _items(20)

    resolved_train, resolved_valid, source = _resolve_validation(train, [], seed=42)

    assert source == "holdout_from_train"
    assert resolved_valid, "a holdout must be produced"
    train_ids = {item["id"] for item in resolved_train}
    valid_ids = {item["id"] for item in resolved_valid}
    assert not (train_ids & valid_ids), "holdout must not leak back into train"
    assert train_ids | valid_ids == {item["id"] for item in train}


def test_holdout_is_deterministic_for_a_seed():
    train = _items(20)

    first = _resolve_validation(train, [], seed=42)[1]
    second = _resolve_validation(train, [], seed=42)[1]

    assert [item["id"] for item in first] == [item["id"] for item in second]


def test_holdout_is_stratified_when_a_class_key_is_given():
    train = _items(30, class_ids=[0] * 10 + [1] * 10 + [2] * 10)

    _, valid, source = _resolve_validation(
        train, [], seed=42, stratify_key=lambda item: item["class_id"]
    )

    assert source == "holdout_from_train"
    assert {item["class_id"] for item in valid} == {0, 1, 2}


def test_tiny_training_set_reports_that_it_reused_train():
    """With one example there is nothing to hold back — say so, don't pretend."""
    train = _items(1)

    _, valid, source = _resolve_validation(train, [], seed=42)

    assert source == "train_split_reused"
    assert valid == train


def test_batches_are_reshuffled_and_cover_every_item():
    batches = _shuffled_batches(10, 4, random.Random(42))

    assert [len(batch) for batch in batches] == [4, 4, 2]
    assert sorted(index for batch in batches for index in batch) == list(range(10))


def test_shuffling_breaks_label_grouped_file_order():
    """Items arrive sorted by filename, which groups them by label.

    Unshuffled, every batch is single-class and the gradient is dominated by
    whichever label sorts first.
    """
    class_ids = [0] * 8 + [1] * 8
    batches = _shuffled_batches(16, 4, random.Random(42))

    mixed = [batch for batch in batches if len({class_ids[index] for index in batch}) > 1]
    assert mixed, "expected at least one batch to contain more than one class"


def test_shuffle_is_deterministic_for_a_seed():
    assert _shuffled_batches(12, 5, random.Random(7)) == _shuffled_batches(12, 5, random.Random(7))
