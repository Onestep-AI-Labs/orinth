from app.ml.common.catalog import TrainingModelDefinition

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
)

ULTRALYTICS_MODEL_OPTIONS = [
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
        )
        for item in ULTRALYTICS_MODEL_OPTIONS
    )
    return options
