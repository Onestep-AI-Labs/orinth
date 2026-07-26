from typing import Any

from app.ml.common.advanced import AUGMENTATION, OPTIMIZATION, RUNTIME, number, toggle
from app.ml.common.catalog import TrainingModelDefinition

# Keys map to Ultralytics `model.train(**kwargs)` arguments (see the YOLO
# runner). Learning rate, optimizer, epochs, image size, batch size and
# device stay as basic top-level fields, so they are deliberately absent here
# rather than double-declared.
YOLO_ADVANCED_PARAMETERS = [
    number("lrf", "Final LR fraction", default=0.15, group=OPTIMIZATION, minimum=0.001, maximum=1.0, step=0.01, help="Final learning rate as a fraction of the initial rate."),
    number("momentum", "Momentum", default=0.9, group=OPTIMIZATION, minimum=0.6, maximum=0.999, step=0.001),
    number("weight_decay", "Weight decay", default=0.0005, group=OPTIMIZATION, minimum=0.0, maximum=0.01, step=0.0005),
    number("warmup_epochs", "Warmup epochs", default=5, group=OPTIMIZATION, minimum=0, maximum=20, integer=True),
    number("close_mosaic", "Close mosaic (epochs)", default=25, group=OPTIMIZATION, minimum=0, maximum=50, integer=True, help="Disable mosaic augmentation for the final N epochs."),
    toggle("cos_lr", "Cosine LR schedule", default=False, group=OPTIMIZATION),
    number("seed", "Seed", default=0, group=RUNTIME, minimum=0, maximum=1_000_000, integer=True, help="Determinism is best-effort on GPU."),
    number("mosaic", "Mosaic", default=0.5, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.05),
    number("mixup", "Mixup", default=0.1, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.05),
    number("fliplr", "Flip horizontal", default=0.5, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.05),
    number("flipud", "Flip vertical", default=0.0, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.05),
    number("hsv_h", "HSV hue", default=0.005, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.005),
    number("hsv_s", "HSV saturation", default=0.5, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.05),
    number("hsv_v", "HSV value", default=0.3, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.05),
    number("degrees", "Rotation degrees", default=5.0, group=AUGMENTATION, minimum=0.0, maximum=45.0, step=1.0),
    number("translate", "Translate", default=0.05, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.05),
    number("scale", "Scale", default=0.2, group=AUGMENTATION, minimum=0.0, maximum=1.0, step=0.05),
]

YOLO_LOCAL_OPTION = TrainingModelDefinition(
    id="yolo_local",
    name="Local YOLO segmentation weights",
    family="yolo",
    task_types=["object_detection", "segmentation"],
    source="local",
    runnable=True,
    needs_download=False,
    description="Fine-tune from the local YOLO weights already referenced by the project.",
    defaults={
        "epochs": 50,
        "image_size": 512,
        "optimizer": "AdamW",
        "learning_rate": 0.002,
        "weights": "local",
    },
    advanced_parameters=YOLO_ADVANCED_PARAMETERS,
)

ULTRALYTICS_MODEL_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "ultralytics_yolo11_detect",
        "name": "Ultralytics YOLO11 Detection",
        "family": "yolo",
        "task_types": ["object_detection"],
        "weights": "yolo11n.pt",
        "runnable": True,
        "needs_download": True,
        "description": "YOLO11 nano detection fine-tuning through the Ultralytics YOLO runner.",
    },
    {
        "id": "ultralytics_yolo11_segment",
        "name": "Ultralytics YOLO11 Segmentation",
        "family": "yolo",
        "task_types": ["segmentation"],
        "weights": "yolo11n-seg.pt",
        "runnable": True,
        "needs_download": True,
        "description": "YOLO11 nano instance segmentation fine-tuning through the Ultralytics YOLO runner.",
    },
    {
        "id": "ultralytics_yolo26_detect",
        "name": "Ultralytics YOLO26 Detection",
        "family": "yolo",
        "task_types": ["object_detection"],
        "weights": "yolo26n.pt",
        "runnable": True,
        "needs_download": True,
        "description": "YOLO26 nano detection fine-tuning through the Ultralytics YOLO runner.",
    },
    {
        "id": "ultralytics_yolo26_segment",
        "name": "Ultralytics YOLO26 Segmentation",
        "family": "yolo",
        "task_types": ["segmentation"],
        "weights": "yolo26n-seg.pt",
        "runnable": True,
        "needs_download": True,
        "description": "YOLO26 nano instance segmentation fine-tuning through the Ultralytics YOLO runner.",
    },
    {
        "id": "ultralytics_sam3",
        "name": "Ultralytics SAM3",
        "family": "sam",
        "task_types": ["segmentation"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "Promptable segmentation catalog option. Training is gated until the dataset workflow is validated.",
    },
    {
        "id": "ultralytics_mobilesam",
        "name": "Ultralytics MobileSAM",
        "family": "sam",
        "task_types": ["segmentation"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "Mobile promptable segmentation catalog option, gated pending validation.",
    },
    {
        "id": "ultralytics_fastsam",
        "name": "Ultralytics FastSAM",
        "family": "sam",
        "task_types": ["segmentation"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "Fast promptable segmentation catalog option, gated pending validation.",
    },
    {
        "id": "ultralytics_yolo_nas",
        "name": "Ultralytics YOLO-NAS",
        "family": "yolo_nas",
        "task_types": ["object_detection"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "YOLO-NAS detection catalog option, gated until runner support is validated.",
    },
    {
        "id": "ultralytics_rt_detr",
        "name": "Ultralytics RT-DETR",
        "family": "rt_detr",
        "task_types": ["object_detection"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "RT-DETR detection catalog option, gated until runner support is validated.",
    },
    {
        "id": "ultralytics_yolo_world",
        "name": "Ultralytics YOLO-World",
        "family": "yolo_world",
        "task_types": ["object_detection"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "Open-vocabulary detection catalog option, gated pending validation.",
    },
]

ULTRALYTICS_OPTIONS_BY_ID = {item["id"]: item for item in ULTRALYTICS_MODEL_OPTIONS}


def yolo_training_options() -> list[TrainingModelDefinition]:
    options = [YOLO_LOCAL_OPTION]
    options.extend(
        TrainingModelDefinition(
            id=item["id"],
            name=item["name"],
            family=item["family"],
            task_types=item["task_types"],
            source="ultralytics",
            runnable=item["runnable"],
            needs_download=item["needs_download"],
            description=item["description"],
            defaults={
                "epochs": 50,
                "image_size": 640,
                "optimizer": "AdamW",
                "learning_rate": 0.002,
                "weights": item["weights"],
                "loader": "YOLO" if item["family"] == "yolo" else item["family"],
            },
            # Only the runnable YOLO family consumes these; gated SAM/NAS/etc.
            # entries stay bare until their runner support is validated.
            advanced_parameters=(
                YOLO_ADVANCED_PARAMETERS if item["family"] == "yolo" and item["runnable"] else []
            ),
        )
        for item in ULTRALYTICS_MODEL_OPTIONS
    )
    return options
