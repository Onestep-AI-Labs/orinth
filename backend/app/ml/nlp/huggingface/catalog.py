from typing import Any

from app.ml.common.catalog import TrainingModelDefinition

HUGGINGFACE_MODEL_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "hf_bert_text_classifier",
        "name": "Hugging Face BERT Classifier",
        "family": "hf_bert_text_classification",
        "task_types": ["text_classification"],
        "model_id": "bert-base-uncased",
        "description": "Fine-tune BERT with a sequence classification head.",
        "defaults": {
            "epochs": 3,
            "optimizer": "adamw",
            "learning_rate": 0.00002,
            "batch_size": 8,
            "max_length": 256,
        },
    },
    {
        "id": "hf_bert_question_answering",
        "name": "Hugging Face BERT QA",
        "family": "hf_bert_question_answering",
        "task_types": ["question_answering"],
        "model_id": "bert-base-uncased",
        "description": "Fine-tune BERT with an extractive question-answering span head.",
        "defaults": {
            "epochs": 3,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 4,
            "max_length": 384,
        },
    },
    {
        "id": "hf_bart_summarizer",
        "name": "Hugging Face BART Summarizer",
        "family": "hf_bart_summarization",
        "task_types": ["summarization"],
        "model_id": "facebook/bart-base",
        "description": "Fine-tune BART for sequence-to-sequence summarization.",
        "defaults": {
            "epochs": 3,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 2,
            "max_length": 512,
            "target_max_length": 96,
        },
    },
]

HUGGINGFACE_OPTIONS_BY_ID = {item["id"]: item for item in HUGGINGFACE_MODEL_OPTIONS}


def huggingface_training_options() -> list[TrainingModelDefinition]:
    return [
        TrainingModelDefinition(
            id=item["id"],
            name=item["name"],
            family=item["family"],
            task_types=item["task_types"],
            source="huggingface",
            runnable=True,
            needs_download=True,
            description=item["description"],
            defaults={**item["defaults"], "model_id": item["model_id"]},
        )
        for item in HUGGINGFACE_MODEL_OPTIONS
    ]


def bert_training_options() -> list[TrainingModelDefinition]:
    return [
        item
        for item in huggingface_training_options()
        if item.id in {"hf_bert_text_classifier", "hf_bert_question_answering"}
    ]


def bart_training_options() -> list[TrainingModelDefinition]:
    return [item for item in huggingface_training_options() if item.id == "hf_bart_summarizer"]


def huggingface_model_id(option_id: str) -> str:
    option = HUGGINGFACE_OPTIONS_BY_ID.get(option_id)
    return str(option["model_id"]) if option else option_id
