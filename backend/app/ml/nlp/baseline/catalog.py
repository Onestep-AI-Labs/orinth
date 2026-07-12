from app.ml.common.catalog import TrainingModelDefinition


def baseline_nlp_training_options() -> list[TrainingModelDefinition]:
    return [
        TrainingModelDefinition(
            id="nlp_tfidf_classifier",
            name="TF-IDF Logistic Regression",
            family="nlp_text_classification",
            task_types=["text_classification"],
            source="local",
            runnable=True,
            needs_download=False,
            description="Fast TF-IDF + Logistic Regression baseline for text classification.",
            defaults={
                "epochs": 5,
                "optimizer": "liblinear",
                "learning_rate": 1.0,
                "batch_size": 0,
            },
        ),
        TrainingModelDefinition(
            id="nlp_extractive_summarizer",
            name="Extractive Summarizer",
            family="nlp_summarization",
            task_types=["summarization"],
            source="local",
            runnable=True,
            needs_download=False,
            description="Offline extractive summarization baseline using reference-summary keywords.",
            defaults={"epochs": 1, "optimizer": "keyword", "learning_rate": 1.0, "batch_size": 0},
        ),
        TrainingModelDefinition(
            id="nlp_keyword_qa",
            name="Keyword QA",
            family="nlp_qa",
            task_types=["question_answering"],
            source="local",
            runnable=True,
            needs_download=False,
            description="Offline question-answering baseline using question/context keyword overlap.",
            defaults={"epochs": 1, "optimizer": "keyword", "learning_rate": 1.0, "batch_size": 0},
        ),
    ]
