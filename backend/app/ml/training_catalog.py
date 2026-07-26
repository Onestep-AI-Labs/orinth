from app.ml.common.catalog import TrainingModelDefinition
from app.ml.llm.catalog import llm_training_options
from app.ml.nlp.baseline.catalog import baseline_nlp_training_options
from app.ml.nlp.huggingface.catalog import huggingface_training_options
from app.ml.nlp.keras_bilstm.catalog import keras_bilstm_training_options
from app.ml.nlp.keras_cnn.catalog import keras_cnn_training_options
from app.ml.nlp.keras_lstm.catalog import keras_lstm_training_options
from app.ml.nlp.seq2seq.catalog import keras_seq2seq_training_options
from app.ml.vision.keras_classification.catalog import keras_classification_training_options
from app.ml.vision.yolo.catalog import yolo_training_options
from app.schemas import TrainingModelOption


def training_model_definitions() -> list[TrainingModelDefinition]:
    definitions = []
    definitions.extend(yolo_training_options())
    definitions.extend(keras_classification_training_options())
    definitions.extend(
        [
            TrainingModelDefinition(
                id="hf_vit_base",
                name="Hugging Face ViT Base",
                family="transformer_classification",
                task_types=["classification"],
                source="huggingface",
                runnable=False,
                needs_download=True,
                description="Catalog entry for future transformer-based image classification training.",
                defaults={"epochs": 5, "image_size": 224, "optimizer": "adamw"},
            )
        ]
    )
    definitions.extend(keras_cnn_training_options())
    definitions.extend(keras_lstm_training_options())
    definitions.extend(keras_bilstm_training_options())
    definitions.extend(keras_seq2seq_training_options())
    definitions.extend(huggingface_training_options())
    definitions.extend(baseline_nlp_training_options())
    definitions.extend(llm_training_options())
    return definitions


def training_model_options(task_type: str | None = None) -> list[TrainingModelOption]:
    options = [definition.to_option() for definition in training_model_definitions()]
    if task_type:
        options = [option for option in options if task_type in option.task_types]
    return options
