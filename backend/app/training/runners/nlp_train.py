"""Stable subprocess entrypoint for NLP training runs.

Invoked as ``python -m app.training.runners.nlp_train``; dispatches to the
per-model-family implementations under ``app.training.runners.nlp``.
"""

import argparse
import json
from pathlib import Path

from app.ml.nlp.huggingface.catalog import (
    CLASSIFICATION_FAMILY,
    QA_FAMILY,
    SUMMARIZATION_FAMILY,
    huggingface_family,
    huggingface_model_id,
)
from app.training.runners.advanced import parse_advanced
from app.training.runners.nlp.baseline import train_qa, train_summarizer, train_text_classifier
from app.training.runners.nlp.common import load_labels
from app.training.runners.nlp.huggingface import (
    train_hf_qa,
    train_hf_summarizer,
    train_hf_text_classifier,
)
from app.training.runners.nlp.keras import (
    train_keras_seq2seq_summarizer,
    train_keras_text_classifier,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--model-option-id", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--target-max-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--hf-model-id", default="")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--advanced", default="{}")
    args = parser.parse_args()

    advanced = parse_advanced(args.advanced)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    labels = load_labels(dataset_root)

    model_option_id = args.model_option_id or "nlp_tfidf_classifier"
    # Hugging Face options dispatch by family so adding a checkpoint to the
    # catalog does not also require a branch here.
    hf_family = huggingface_family(model_option_id)
    if args.task_type == "text_classification" and model_option_id == "nlp_tfidf_classifier":
        metrics, predictions = train_text_classifier(
            dataset_root, run_dir, labels, max(args.epochs, 1), args.learning_rate, advanced=advanced
        )
    elif args.task_type == "text_classification" and model_option_id in {
        "nlp_keras_cnn_classifier",
        "nlp_keras_lstm_classifier",
        "nlp_keras_bilstm_classifier",
    }:
        metrics, predictions = train_keras_text_classifier(
            dataset_root=dataset_root,
            run_dir=run_dir,
            labels=labels,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 8),
            vocab_size=max(args.vocab_size, 100),
            model_kind=model_option_id.replace("nlp_keras_", "").replace("_classifier", ""),
        )
    elif hf_family == CLASSIFICATION_FAMILY:
        metrics, predictions = train_hf_text_classifier(
            dataset_root=dataset_root,
            run_dir=run_dir,
            labels=labels,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 8),
            model_id=args.hf_model_id or huggingface_model_id(model_option_id),
            cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
            advanced=advanced,
        )
    elif args.task_type == "summarization" and model_option_id == "nlp_extractive_summarizer":
        metrics, predictions = train_summarizer(dataset_root, run_dir, max(args.epochs, 1))
    elif args.task_type == "summarization" and model_option_id == "nlp_keras_seq2seq_summarizer":
        metrics, predictions = train_keras_seq2seq_summarizer(
            dataset_root=dataset_root,
            run_dir=run_dir,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 16),
            target_max_length=max(args.target_max_length, 8),
            vocab_size=max(args.vocab_size, 100),
        )
    elif hf_family == SUMMARIZATION_FAMILY:
        metrics, predictions = train_hf_summarizer(
            dataset_root=dataset_root,
            run_dir=run_dir,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 16),
            target_max_length=max(args.target_max_length, 8),
            model_id=args.hf_model_id or huggingface_model_id(model_option_id),
            cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
            advanced=advanced,
        )
    elif args.task_type == "question_answering" and model_option_id == "nlp_keyword_qa":
        metrics, predictions = train_qa(dataset_root, run_dir, max(args.epochs, 1))
    elif hf_family == QA_FAMILY:
        metrics, predictions = train_hf_qa(
            dataset_root=dataset_root,
            run_dir=run_dir,
            epochs=max(args.epochs, 1),
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 1),
            max_length=max(args.max_length, 64),
            model_id=args.hf_model_id or huggingface_model_id(model_option_id),
            cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
            advanced=advanced,
        )
    else:
        raise ValueError(f"Unsupported NLP task/model option: {args.task_type}/{model_option_id}")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "validation_predictions.json").write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    print(f"NLP training finished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
