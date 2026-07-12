from app.ml.common.catalog import TrainingModelDefinition


def keras_bilstm_training_options() -> list[TrainingModelDefinition]:
    return [
        TrainingModelDefinition(
            id="nlp_keras_bilstm_classifier",
            name="Keras Text BiLSTM",
            family="nlp_keras_bilstm",
            task_types=["text_classification"],
            source="local",
            runnable=True,
            needs_download=False,
            description="Train a local Keras embedding + bidirectional LSTM classifier from scratch.",
            defaults={
                "epochs": 8,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 16,
                "max_length": 160,
                "vocab_size": 12000,
            },
        )
    ]
