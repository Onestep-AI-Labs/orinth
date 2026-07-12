from app.ml.common.catalog import TrainingModelDefinition


def keras_seq2seq_training_options() -> list[TrainingModelDefinition]:
    return [
        TrainingModelDefinition(
            id="nlp_keras_seq2seq_summarizer",
            name="Keras Seq2Seq Summarizer",
            family="nlp_keras_seq2seq",
            task_types=["summarization"],
            source="local",
            runnable=True,
            needs_download=False,
            description="Train a local encoder-decoder LSTM summarizer from scratch.",
            defaults={
                "epochs": 8,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 8,
                "max_length": 192,
                "target_max_length": 48,
                "vocab_size": 16000,
            },
        )
    ]
