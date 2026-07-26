from app.ml.common.advanced import OPTIMIZATION, REGULARIZATION, number, select
from app.ml.common.catalog import TrainingModelDefinition

# Keys map to the TF-IDF + Logistic Regression baseline runner.
SKLEARN_ADVANCED_PARAMETERS = [
    number("tfidf_max_features", "TF-IDF max features", default=20000, group=OPTIMIZATION, minimum=1000, maximum=100000, integer=True, help="0 or empty keeps the full vocabulary."),
    select("tfidf_ngram_range", "TF-IDF n-gram range", options=["1,1", "1,2", "1,3", "2,2"], default="1,2", group=OPTIMIZATION, help="Lower and upper word n-gram bounds."),
    number("C", "Regularisation (C)", default=1.0, group=OPTIMIZATION, minimum=0.001, maximum=100.0, step=0.1, help="Inverse regularisation strength; overrides the basic field."),
    number("max_iter", "Max iterations", default=1000, group=OPTIMIZATION, minimum=100, maximum=10000, integer=True),
    select("class_weight", "Class weight", options=["none", "balanced"], default="none", group=REGULARIZATION),
]


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
            advanced_parameters=SKLEARN_ADVANCED_PARAMETERS,
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
