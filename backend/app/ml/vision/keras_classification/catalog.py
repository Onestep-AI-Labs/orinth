from typing import Any

from app.ml.common.advanced import (
    AUGMENTATION,
    OPTIMIZATION,
    REGULARIZATION,
    RUNTIME,
    number,
    select,
    toggle,
)
from app.ml.common.catalog import TrainingModelDefinition

# Keys map to the Keras classification runner. Optimizer, learning rate,
# epochs, image size and batch size stay as basic top-level fields.
KERAS_ADVANCED_PARAMETERS = [
    select("lr_schedule", "LR schedule", options=["constant", "cosine", "step", "plateau"], default="constant", group=OPTIMIZATION),
    number("unfreeze_layers", "Unfreeze top layers", default=0, group=OPTIMIZATION, minimum=0, maximum=200, integer=True, help="Fine-tune the last N backbone layers; 0 keeps the backbone frozen."),
    number("early_stop_patience", "Early stop patience", default=0, group=OPTIMIZATION, minimum=0, maximum=50, integer=True, help="Stop after N epochs without val-accuracy gain; 0 disables."),
    number("seed", "Seed", default=42, group=RUNTIME, minimum=0, maximum=1_000_000, integer=True, help="Determinism is best-effort on GPU."),
    number("dropout", "Dropout", default=0.2, group=REGULARIZATION, minimum=0.0, maximum=0.9, step=0.05),
    number("label_smoothing", "Label smoothing", default=0.0, group=REGULARIZATION, minimum=0.0, maximum=0.3, step=0.01),
    toggle("class_weighting", "Balance class weights", default=False, group=REGULARIZATION),
    toggle("aug_horizontal_flip", "Random horizontal flip", default=False, group=AUGMENTATION),
    toggle("aug_rotation", "Random rotation", default=False, group=AUGMENTATION),
    toggle("aug_zoom", "Random zoom", default=False, group=AUGMENTATION),
    toggle("aug_contrast", "Random contrast", default=False, group=AUGMENTATION),
]

KERAS_APPLICATION_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "keras_mobilenet_v2",
        "name": "Keras MobileNetV2",
        "app_name": "MobileNetV2",
        "image_size": 224,
        "kwargs": {"alpha": 1.0},
        "description": "Lightweight ImageNet CNN suitable for fast transfer learning.",
    },
    {
        "id": "keras_efficientnet_b0",
        "name": "Keras EfficientNetB0",
        "app_name": "EfficientNetB0",
        "image_size": 224,
        "kwargs": {},
        "description": "Balanced EfficientNet baseline for image classification transfer learning.",
    },
    {
        "id": "keras_efficientnet_b1",
        "name": "Keras EfficientNetB1",
        "app_name": "EfficientNetB1",
        "image_size": 240,
        "kwargs": {},
        "description": "EfficientNet B1 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b2",
        "name": "Keras EfficientNetB2",
        "app_name": "EfficientNetB2",
        "image_size": 260,
        "kwargs": {},
        "description": "EfficientNet B2 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b3",
        "name": "Keras EfficientNetB3",
        "app_name": "EfficientNetB3",
        "image_size": 300,
        "kwargs": {},
        "description": "EfficientNet B3 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b4",
        "name": "Keras EfficientNetB4",
        "app_name": "EfficientNetB4",
        "image_size": 380,
        "kwargs": {},
        "description": "EfficientNet B4 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b5",
        "name": "Keras EfficientNetB5",
        "app_name": "EfficientNetB5",
        "image_size": 456,
        "kwargs": {},
        "description": "EfficientNet B5 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b6",
        "name": "Keras EfficientNetB6",
        "app_name": "EfficientNetB6",
        "image_size": 528,
        "kwargs": {},
        "description": "EfficientNet B6 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b7",
        "name": "Keras EfficientNetB7",
        "app_name": "EfficientNetB7",
        "image_size": 600,
        "kwargs": {"name": "efficientnetb7"},
        "description": "Large EfficientNet B7 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_v2_b0",
        "name": "Keras EfficientNetV2B0",
        "app_name": "EfficientNetV2B0",
        "image_size": 224,
        "kwargs": {},
        "description": "EfficientNetV2 B0 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_v2_b3",
        "name": "Keras EfficientNetV2B3",
        "app_name": "EfficientNetV2B3",
        "image_size": 300,
        "kwargs": {},
        "description": "EfficientNetV2 B3 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_v2_s",
        "name": "Keras EfficientNetV2S",
        "app_name": "EfficientNetV2S",
        "image_size": 384,
        "kwargs": {},
        "description": "EfficientNetV2 S ImageNet backbone.",
    },
    {
        "id": "keras_resnet50",
        "name": "Keras ResNet50",
        "app_name": "ResNet50",
        "image_size": 224,
        "kwargs": {},
        "description": "Classic ResNet50 ImageNet backbone.",
    },
    {
        "id": "keras_xception",
        "name": "Keras Xception",
        "app_name": "Xception",
        "image_size": 299,
        "kwargs": {},
        "description": "Xception ImageNet backbone.",
    },
    {
        "id": "keras_inception_v3",
        "name": "Keras InceptionV3",
        "app_name": "InceptionV3",
        "image_size": 299,
        "kwargs": {},
        "description": "InceptionV3 ImageNet backbone.",
    },
    {
        "id": "keras_densenet121",
        "name": "Keras DenseNet121",
        "app_name": "DenseNet121",
        "image_size": 224,
        "kwargs": {},
        "description": "DenseNet121 ImageNet backbone.",
    },
    {
        "id": "keras_convnext_tiny",
        "name": "Keras ConvNeXtTiny",
        "app_name": "ConvNeXtTiny",
        "image_size": 224,
        "kwargs": {},
        "description": "ConvNeXt Tiny ImageNet backbone.",
    },
]

KERAS_APPLICATIONS_BY_ID = {item["id"]: item for item in KERAS_APPLICATION_OPTIONS}


def keras_classification_training_options() -> list[TrainingModelDefinition]:
    return [
        TrainingModelDefinition(
            id=item["id"],
            name=item["name"],
            family="keras_classification",
            task_types=["classification"],
            source="keras_applications",
            runnable=True,
            needs_download=True,
            description=item["description"],
            defaults={
                "epochs": 10,
                "image_size": item["image_size"],
                "optimizer": "adam",
                "weights": "imagenet",
                "include_top": False,
                "pooling": "avg",
                "base_model": item["app_name"],
                "application_kwargs": item["kwargs"],
            },
            advanced_parameters=KERAS_ADVANCED_PARAMETERS,
        )
        for item in KERAS_APPLICATION_OPTIONS
    ]
