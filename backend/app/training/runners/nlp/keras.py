"""Keras-based NLP model families: text classification and seq2seq summarization."""

import json
from pathlib import Path
from typing import Any

from app.training.runners.nlp.common import (
    average_scores,
    load_summary_split,
    load_text_classification_split,
    write_keras_history_csv,
)


def train_keras_text_classifier(
    *,
    dataset_root: Path,
    run_dir: Path,
    labels: list[str],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    vocab_size: int,
    model_kind: str,
) -> tuple[dict, list[dict]]:
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_text_classification_split(dataset_root, "train")
    valid = load_text_classification_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Keras text classification training requires labeled train texts")

    train_texts = [item["text"] for item in train]
    valid_texts = [item["text"] for item in valid]
    train_y = np.asarray([int(item["class_id"]) for item in train], dtype="int32")
    valid_y = np.asarray([int(item["class_id"]) for item in valid], dtype="int32")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    x_train = _keras_pad_texts(tf, tokenizer, train_texts, max_length)
    x_valid = _keras_pad_texts(tf, tokenizer, valid_texts, max_length)

    model = _build_keras_text_classifier(
        tf=tf,
        model_kind=model_kind,
        vocab_size=vocab_size,
        max_length=max_length,
        num_labels=len(labels),
        learning_rate=learning_rate,
    )
    history = model.fit(
        x_train,
        train_y,
        validation_data=(x_valid, valid_y),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    model.save(run_dir / "best_model.keras")
    model.save(run_dir / "last_model.keras")
    (run_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")

    probabilities = model.predict(x_valid, verbose=0)
    y_pred = [int(value) for value in np.argmax(probabilities, axis=1)]
    y_true = [int(value) for value in valid_y.tolist()]
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics = {
        "classes": labels,
        "model_kind": model_kind,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    metadata = {
        "artifact_type": "keras_text_classifier",
        "model_kind": model_kind,
        "labels": labels,
        "max_length": max_length,
        "vocab_size": vocab_size,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    predictions = []
    for index, item in enumerate(valid):
        scores = {
            labels[class_index]: float(probabilities[index][class_index])
            for class_index in range(min(len(labels), probabilities.shape[1]))
        }
        predictions.append(
            {
                "item": item["id"],
                "ground_truth": labels[y_true[index]] if y_true[index] < len(labels) else str(y_true[index]),
                "prediction": labels[y_pred[index]] if y_pred[index] < len(labels) else str(y_pred[index]),
                "scores": scores,
            }
        )
    write_keras_history_csv(run_dir / "results.csv", history.history)
    return metrics, predictions


def train_keras_seq2seq_summarizer(
    *,
    dataset_root: Path,
    run_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    target_max_length: int,
    vocab_size: int,
) -> tuple[dict, list[dict]]:
    import numpy as np
    import tensorflow as tf

    from app.services.metrics import rouge_scores

    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_summary_split(dataset_root, "train")
    valid = load_summary_split(dataset_root, "valid") or train
    if not train:
        raise ValueError("Keras seq2seq summarization training requires reference summaries")

    start_token = "startseq"
    end_token = "endseq"
    train_sources = [item["text"] for item in train]
    train_targets = [f"{start_token} {item['summary']} {end_token}" for item in train]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        filters="",
        lower=True,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts([*train_sources, *train_targets])
    encoder_train = _keras_pad_texts(tf, tokenizer, train_sources, max_length)
    target_sequences = tokenizer.texts_to_sequences(train_targets)
    decoder_input = tf.keras.preprocessing.sequence.pad_sequences(
        [sequence[:-1] for sequence in target_sequences],
        maxlen=target_max_length,
        padding="post",
        truncating="post",
    )
    decoder_target = tf.keras.preprocessing.sequence.pad_sequences(
        [sequence[1:] for sequence in target_sequences],
        maxlen=target_max_length,
        padding="post",
        truncating="post",
    )

    model = _build_keras_seq2seq(
        tf=tf,
        vocab_size=vocab_size,
        input_max_length=max_length,
        target_max_length=target_max_length,
        learning_rate=learning_rate,
    )
    history = model.fit(
        [encoder_train, decoder_input],
        np.expand_dims(decoder_target, axis=-1),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    model.save(run_dir / "best_model.keras")
    model.save(run_dir / "last_model.keras")
    (run_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    metadata = {
        "artifact_type": "keras_seq2seq_summarizer",
        "labels": ["summary"],
        "max_length": max_length,
        "target_max_length": target_max_length,
        "vocab_size": vocab_size,
        "start_token": start_token,
        "end_token": end_token,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = []
    predictions = []
    for item in valid:
        prediction = _keras_seq2seq_generate(
            tf=tf,
            model=model,
            tokenizer=tokenizer,
            text=item["text"],
            max_length=max_length,
            target_max_length=target_max_length,
            start_token=start_token,
            end_token=end_token,
        )
        score = rouge_scores(prediction, item["summary"])
        rows.append(score)
        predictions.append(
            {
                "item": item["id"],
                "reference": item["summary"],
                "prediction": prediction,
                "scores": score,
            }
        )
    metrics: dict[str, Any] = average_scores(rows)
    metrics["model_kind"] = "keras_seq2seq"
    write_keras_history_csv(run_dir / "results.csv", history.history)
    return metrics, predictions


def _keras_pad_texts(tf, tokenizer, texts: list[str], max_length: int):
    sequences = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )


def _build_keras_text_classifier(
    *,
    tf,
    model_kind: str,
    vocab_size: int,
    max_length: int,
    num_labels: int,
    learning_rate: float,
):
    inputs = tf.keras.Input(shape=(max_length,), dtype="int32")
    x = tf.keras.layers.Embedding(vocab_size, 64)(inputs)
    if model_kind == "cnn":
        x = tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    elif model_kind == "lstm":
        x = tf.keras.layers.LSTM(64)(x)
    elif model_kind == "bilstm":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    else:
        raise ValueError(f"Unsupported Keras text model kind: {model_kind}")
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_keras_seq2seq(
    *,
    tf,
    vocab_size: int,
    input_max_length: int,
    target_max_length: int,
    learning_rate: float,
):
    encoder_inputs = tf.keras.Input(shape=(input_max_length,), dtype="int32", name="encoder_input")
    decoder_inputs = tf.keras.Input(shape=(target_max_length,), dtype="int32", name="decoder_input")
    encoder_embedding = tf.keras.layers.Embedding(vocab_size, 96, name="encoder_embedding")(encoder_inputs)
    _encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
        96,
        return_state=True,
        name="encoder_lstm",
    )(encoder_embedding)
    decoder_embedding = tf.keras.layers.Embedding(vocab_size, 96, name="decoder_embedding")(decoder_inputs)
    decoder_outputs = tf.keras.layers.LSTM(
        96,
        return_sequences=True,
        name="decoder_lstm",
    )(decoder_embedding, initial_state=[state_h, state_c])
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", name="token_output")(decoder_outputs)
    model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
    )
    return model


def _keras_seq2seq_generate(
    *,
    tf,
    model,
    tokenizer,
    text: str,
    max_length: int,
    target_max_length: int,
    start_token: str,
    end_token: str,
) -> str:
    source = _keras_pad_texts(tf, tokenizer, [text], max_length)
    start_id = int(tokenizer.word_index.get(start_token, 0))
    end_id = int(tokenizer.word_index.get(end_token, 0))
    decoder = [start_id]
    for _ in range(target_max_length):
        decoder_input = tf.keras.preprocessing.sequence.pad_sequences(
            [decoder],
            maxlen=target_max_length,
            padding="post",
            truncating="post",
        )
        prediction = model.predict([source, decoder_input], verbose=0)[0]
        next_position = min(len(decoder) - 1, prediction.shape[0] - 1)
        next_id = int(prediction[next_position].argmax())
        if next_id in {0, end_id}:
            break
        decoder.append(next_id)
    words = [
        tokenizer.index_word.get(token_id, "")
        for token_id in decoder[1:]
        if token_id not in {0, start_id, end_id}
    ]
    return " ".join(word for word in words if word).strip()
