from app.ml.common.catalog import TrainingModelDefinition


def keras_lstm_training_options() -> list[TrainingModelDefinition]:
    return [
        TrainingModelDefinition(
            id="nlp_keras_lstm_classifier",
            name="Keras Text LSTM",
            family="nlp_keras_lstm",
            task_types=["text_classification"],
            source="local",
            runnable=True,
            needs_download=False,
            description="Train a local Keras embedding + LSTM classifier from scratch.",
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
