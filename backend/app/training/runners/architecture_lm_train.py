"""Train a from-scratch language model built in the architecture studio.

Phase 17 stage E. The graph emits a Keras model whose output is
`(sequence, vocabulary)` logits; this runner builds a vocabulary from a text
corpus, windows it into next-token pairs, trains, and generates samples.

This is a research surface at toy scale. A transformer trained from random
initialization on a local corpus will not be competitive with a pretrained
base — `specs/phase-14-llm-finetuning.md` remains the path for real LLM work.
What it does give you is a model whose every layer you chose, whose loss you
can watch fall, and whose code you can read.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.ml.architecture.lm_catalog import LM_ADVANCED_PARAMETERS
from app.training.runners.advanced import allowed_keys, log_ignored, parse_advanced, partition
from app.training.runners.architecture_train import load_build_model
from app.training.runners.keras_common import learning_rate_schedule, optimizer_for

LM_ADVANCED_KEYS = allowed_keys(LM_ADVANCED_PARAMETERS)

PAD, UNKNOWN = "<pad>", "<unk>"
WORD_PATTERN = re.compile(r"\w+|[^\w\s]")


def select_advanced(raw: str | None) -> tuple[dict[str, Any], list[str]]:
    return partition(parse_advanced(raw), LM_ADVANCED_KEYS)


# --- corpus and vocabulary --------------------------------------------------


def read_corpus(dataset_root: Path, split: str) -> str:
    """Every `.txt` in a split, concatenated.

    Language modelling needs no annotations — the text is both input and
    target — so this reads the raw files rather than going through the
    annotation-pairing helpers the other NLP runners use.
    """

    text_dir = dataset_root / split / "texts"
    if not text_dir.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(text_dir.glob("*.txt"))
    )


def tokenize(text: str, mode: str) -> list[str]:
    return list(text) if mode == "char" else WORD_PATTERN.findall(text.lower())


def build_vocabulary(tokens: list[str], mode: str, limit: int) -> list[str]:
    """Most-frequent tokens, with pad at index 0 so padding is a real id."""

    counts = Counter(tokens)
    common = [token for token, _count in counts.most_common(max(limit - 2, 1))]
    return [PAD, UNKNOWN, *common]


def encode(tokens: list[str], lookup: dict[str, int]) -> list[int]:
    unknown = lookup[UNKNOWN]
    return [lookup.get(token, unknown) for token in tokens]


def windows(ids: list[int], length: int, stride: int):
    """`(input, next-token)` pairs, sliding over the encoded corpus."""

    import numpy as np

    inputs, targets = [], []
    for start in range(0, len(ids) - length - 1, max(stride, 1)):
        inputs.append(ids[start : start + length])
        targets.append(ids[start + 1 : start + length + 1])
    if not inputs:
        return None, None
    return np.asarray(inputs, dtype="int32"), np.asarray(targets, dtype="int32")


# --- generation -------------------------------------------------------------


def generate(model, prompt: str, vocabulary: list[str], mode: str, *, length: int,
             max_new_tokens: int, temperature: float) -> str:
    """Temperature-sampled continuation, decoded back to text."""

    import numpy as np

    lookup = {token: index for index, token in enumerate(vocabulary)}
    ids = encode(tokenize(prompt, mode), lookup) or [lookup[UNKNOWN]]
    produced: list[int] = []

    for _step in range(max_new_tokens):
        context = ids[-length:]
        padded = [0] * (length - len(context)) + context
        logits = model.predict(np.asarray([padded], dtype="int32"), verbose=0)[0][-1]
        if temperature <= 0:
            choice = int(np.argmax(logits))
        else:
            scaled = logits.astype("float64") / temperature
            probabilities = np.exp(scaled - scaled.max())
            probabilities /= probabilities.sum()
            choice = int(np.random.choice(len(probabilities), p=probabilities))
        ids.append(choice)
        produced.append(choice)

    joiner = "" if mode == "char" else " "
    return joiner.join(vocabulary[token] for token in produced if token < len(vocabulary))


def sample_prompts(text: str, mode: str, count: int = 3) -> list[str]:
    """A few real openings from the corpus, so samples are comparable to it."""

    candidates = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
    if not candidates:
        return [text[:40]] if text else []
    step = max(1, len(candidates) // count)
    picked = candidates[::step][:count]
    return [" ".join(tokenize(line, mode)[:8]) if mode == "word" else line[:40] for line in picked]


# --- entry point ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--advanced", default="{}")
    args = parser.parse_args()

    advanced, ignored = select_advanced(args.advanced)
    log_ignored(ignored)

    import numpy as np
    import tensorflow as tf

    if "seed" in advanced:
        tf.keras.utils.set_random_seed(int(advanced["seed"]))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    mode = str(advanced.get("tokenizer", "word"))
    train_text = read_corpus(dataset_root, "train")
    valid_text = read_corpus(dataset_root, "valid")
    if not train_text.strip():
        raise ValueError(
            "Language-model training needs text files under train/texts in the dataset."
        )
    if not valid_text.strip():
        # Hold out the tail rather than validating on the training text itself.
        split_at = int(len(train_text) * 0.9)
        train_text, valid_text = train_text[:split_at], train_text[split_at:]

    train_tokens = tokenize(train_text, mode)
    vocabulary = build_vocabulary(train_tokens, mode, int(advanced.get("vocab_size", 2000)))
    lookup = {token: index for index, token in enumerate(vocabulary)}
    print(f"Vocabulary: {len(vocabulary)} {mode} tokens from {len(train_tokens)} training tokens")

    build_model = load_build_model(Path(args.model_file))
    model = build_model(num_classes=len(vocabulary))
    model.summary()
    length = sequence_length(model)
    check_output(model, len(vocabulary))

    stride = max(1, int(advanced.get("stride", 0)) or max(1, length // 2))
    x_train, y_train = windows(encode(train_tokens, lookup), length, stride)
    x_valid, y_valid = windows(encode(tokenize(valid_text, mode), lookup), length, stride)
    if x_train is None:
        raise ValueError(
            f"The corpus is shorter than one {length}-token window. Use a longer text "
            "dataset, or lower the Input node's sequence length."
        )
    if x_valid is None:
        x_valid, y_valid = x_train[:1], y_train[:1]
    print(f"Windows: {len(x_train)} train, {len(x_valid)} validation, length {length}")

    steps_per_epoch = max(1, math.ceil(len(x_train) / max(args.batch_size, 1)))
    learning_rate = learning_rate_schedule(
        str(advanced.get("lr_schedule", "cosine")),
        args.learning_rate,
        steps_per_epoch,
        args.epochs,
        tf,
    )
    # The LM head emits raw logits, so the loss applies the softmax itself.
    model.compile(
        optimizer=optimizer_for(args.optimizer, learning_rate, tf),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.CSVLogger(run_dir / "results.csv"),
        tf.keras.callbacks.ModelCheckpoint(
            run_dir / "best_model.keras", monitor="val_loss", mode="min", save_best_only=True
        ),
    ]
    patience = int(advanced.get("early_stop_patience", 0) or 0)
    if patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=patience, restore_best_weights=True
            )
        )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )
    model.save(run_dir / "last_model.keras")

    metrics: dict[str, Any] = {
        key: float(values[-1]) for key, values in history.history.items() if values
    }
    # Perplexity is the number a language model is actually judged on.
    if "val_loss" in metrics:
        metrics["val_perplexity"] = float(np.exp(min(metrics["val_loss"], 20.0)))
    if "loss" in metrics:
        metrics["train_perplexity"] = float(np.exp(min(metrics["loss"], 20.0)))
    metrics["vocab_size"] = len(vocabulary)
    metrics["tokenizer"] = mode
    metrics["sequence_length"] = length
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (run_dir / "tokenizer.json").write_text(
        json.dumps({"mode": mode, "vocabulary": vocabulary}, ensure_ascii=False),
        encoding="utf-8",
    )

    prompts = sample_prompts(valid_text or train_text, mode)
    generations = [
        {
            "prompt": prompt,
            "generated": generate(
                model,
                prompt,
                vocabulary,
                mode,
                length=length,
                max_new_tokens=int(advanced.get("max_new_tokens", 40)),
                temperature=float(advanced.get("temperature", 0.8)),
            ),
        }
        for prompt in prompts
    ]
    (run_dir / "sample_generations.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "keras_lm",
                "tokenizer": mode,
                "vocab_size": len(vocabulary),
                "sequence_length": length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Language-model training finished. Results saved to: {run_dir}")


def sequence_length(model) -> int:
    shape = model.input_shape
    if isinstance(shape, list) or len(shape) != 2 or shape[1] is None:
        raise ValueError(
            "A language model needs a single Input node with a fixed sequence length, "
            "for example 128."
        )
    return int(shape[1])


def check_output(model, vocab_size: int) -> None:
    shape = model.output_shape
    if len(shape) != 3:
        raise ValueError(
            "A language model must output one prediction per position — add an LM head "
            f"as the last layer. This graph outputs {shape[1:]}."
        )
    if shape[-1] != vocab_size:
        raise ValueError(
            f"The LM head emits {shape[-1]} logits but the corpus vocabulary is {vocab_size}. "
            'Turn on "Vocab = dataset vocabulary" on the LM head node.'
        )


if __name__ == "__main__":
    main()
