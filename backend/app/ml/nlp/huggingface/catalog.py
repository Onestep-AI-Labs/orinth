from typing import Any

from app.ml.common.advanced import OPTIMIZATION, RUNTIME, number, select
from app.ml.common.catalog import TrainingModelDefinition

# Keys map to the Hugging Face NLP runner (`runners/nlp/huggingface.py`).
# Epochs, learning rate, batch size and max length stay as basic fields.
HF_ADVANCED_PARAMETERS = [
    number("weight_decay", "Weight decay", default=0.01, group=OPTIMIZATION, minimum=0.0, maximum=0.3, step=0.01),
    number("warmup_ratio", "Warmup ratio", default=0.1, group=OPTIMIZATION, minimum=0.0, maximum=0.5, step=0.01),
    select("lr_scheduler", "LR scheduler", options=["linear", "cosine"], default="linear", group=OPTIMIZATION),
    number("gradient_accumulation_steps", "Gradient accumulation", default=1, group=OPTIMIZATION, minimum=1, maximum=32, integer=True, help="Effective batch size is batch size × this."),
    number("seed", "Seed", default=42, group=RUNTIME, minimum=0, maximum=1_000_000, integer=True),
    select("precision", "Precision", options=["fp32", "fp16", "bf16"], default="fp32", group=RUNTIME, help="fp16/bf16 require CUDA; downgraded to fp32 on MPS/CPU."),
]

CLASSIFICATION_FAMILY = "hf_bert_text_classification"
QA_FAMILY = "hf_bert_question_answering"
SUMMARIZATION_FAMILY = "hf_bart_summarization"

# Ordering is load-bearing: the training page auto-selects the first runnable
# option for a task, so each task lists its best small-data default first.
#
# Every entry reuses one of the three families above — the runners and
# predictors build through `AutoTokenizer` / `AutoModelFor*`, so a new
# checkpoint of the same architecture class needs no new code path.
HUGGINGFACE_MODEL_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "hf_distilbert_text_classifier",
        "name": "DistilBERT Classifier",
        "family": CLASSIFICATION_FAMILY,
        "task_types": ["text_classification"],
        "model_id": "distilbert-base-uncased",
        "description": "Distilled BERT — the fastest of the family and the most forgiving on small datasets.",
        "defaults": {
            "epochs": 10,
            "optimizer": "adamw",
            "learning_rate": 0.00005,
            "batch_size": 16,
            "max_length": 256,
        },
    },
    {
        "id": "hf_roberta_text_classifier",
        "name": "RoBERTa Classifier",
        "family": CLASSIFICATION_FAMILY,
        "task_types": ["text_classification"],
        "model_id": "roberta-base",
        "description": "Robustly optimised BERT pretraining; usually the strongest classifier here.",
        "defaults": {
            "epochs": 10,
            "optimizer": "adamw",
            "learning_rate": 0.00002,
            "batch_size": 8,
            "max_length": 256,
        },
    },
    {
        "id": "hf_bert_text_classifier",
        "name": "BERT Classifier",
        "family": CLASSIFICATION_FAMILY,
        "task_types": ["text_classification"],
        "model_id": "bert-base-uncased",
        "description": "Fine-tune BERT with a sequence classification head.",
        "defaults": {
            "epochs": 10,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 8,
            "max_length": 256,
        },
    },
    {
        "id": "hf_bert_cased_text_classifier",
        "name": "BERT Cased Classifier",
        "family": CLASSIFICATION_FAMILY,
        "task_types": ["text_classification"],
        "model_id": "bert-base-cased",
        "description": "Case-sensitive BERT, for text where capitalisation carries meaning.",
        "defaults": {
            "epochs": 10,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 8,
            "max_length": 256,
        },
    },
    {
        "id": "hf_albert_text_classifier",
        "name": "ALBERT Classifier",
        "family": CLASSIFICATION_FAMILY,
        "task_types": ["text_classification"],
        "model_id": "albert-base-v2",
        "description": "Parameter-shared BERT variant; smallest footprint, needs more epochs to converge.",
        "defaults": {
            "epochs": 14,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 8,
            "max_length": 256,
        },
    },
    {
        "id": "hf_multilingual_bert_text_classifier",
        "name": "Multilingual BERT Classifier",
        "family": CLASSIFICATION_FAMILY,
        "task_types": ["text_classification"],
        "model_id": "bert-base-multilingual-cased",
        "description": "BERT pretrained on 104 languages, for non-English or mixed-language text.",
        "defaults": {
            "epochs": 10,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 8,
            "max_length": 256,
        },
    },
    {
        "id": "hf_distilbert_squad_qa",
        "name": "DistilBERT SQuAD QA",
        "family": QA_FAMILY,
        "task_types": ["question_answering"],
        "model_id": "distilbert-base-cased-distilled-squad",
        "description": "Already fine-tuned for extractive QA, so it answers well before you train it further.",
        "defaults": {
            "epochs": 2,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 8,
            "max_length": 384,
        },
    },
    {
        "id": "hf_roberta_squad_qa",
        "name": "RoBERTa SQuAD2 QA",
        "family": QA_FAMILY,
        "task_types": ["question_answering"],
        "model_id": "deepset/roberta-base-squad2",
        "description": "RoBERTa fine-tuned on SQuAD 2.0; the strongest extractive QA option here.",
        "defaults": {
            "epochs": 2,
            "optimizer": "adamw",
            "learning_rate": 0.00002,
            "batch_size": 8,
            "max_length": 384,
        },
    },
    {
        "id": "hf_bert_question_answering",
        "name": "BERT QA",
        "family": QA_FAMILY,
        "task_types": ["question_answering"],
        "model_id": "bert-base-uncased",
        "description": "Fine-tune BERT with a fresh span head. Needs far more data than a SQuAD-pretrained option.",
        "defaults": {
            "epochs": 4,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 8,
            "max_length": 384,
        },
    },
    {
        "id": "hf_multilingual_bert_qa",
        "name": "Multilingual BERT QA",
        "family": QA_FAMILY,
        "task_types": ["question_answering"],
        "model_id": "bert-base-multilingual-cased",
        "description": "Multilingual BERT with a span head, for non-English question answering.",
        "defaults": {
            "epochs": 4,
            "optimizer": "adamw",
            "learning_rate": 0.00003,
            "batch_size": 8,
            "max_length": 384,
        },
    },
    {
        "id": "hf_bart_summarizer",
        "name": "BART Summarizer",
        "family": SUMMARIZATION_FAMILY,
        "task_types": ["summarization"],
        "model_id": "facebook/bart-base",
        "description": "Fine-tune BART for sequence-to-sequence summarization.",
        "defaults": {
            "epochs": 4,
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
            advanced_parameters=HF_ADVANCED_PARAMETERS,
        )
        for item in HUGGINGFACE_MODEL_OPTIONS
    ]


def bert_training_options() -> list[TrainingModelDefinition]:
    # Filter by family, not by a hardcoded id set: an id list silently drops
    # every checkpoint added to these families later.
    return [
        item
        for item in huggingface_training_options()
        if item.family in {CLASSIFICATION_FAMILY, QA_FAMILY}
    ]


def bart_training_options() -> list[TrainingModelDefinition]:
    return [
        item for item in huggingface_training_options() if item.family == SUMMARIZATION_FAMILY
    ]


def huggingface_model_id(option_id: str) -> str:
    option = HUGGINGFACE_OPTIONS_BY_ID.get(option_id)
    return str(option["model_id"]) if option else option_id


def huggingface_family(option_id: str) -> str | None:
    """Model family for a catalog option id, or ``None`` if it is not ours."""
    option = HUGGINGFACE_OPTIONS_BY_ID.get(option_id)
    return str(option["family"]) if option else None
