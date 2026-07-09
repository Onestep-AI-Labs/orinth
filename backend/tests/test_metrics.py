import numpy as np

from app.schemas import Box, Detection
from app.services.metrics import (
    binary_mask_metrics,
    image_label_from_detections,
    image_label_from_class_ids,
)


def test_binary_mask_metrics_perfect_prediction():
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    metrics = binary_mask_metrics(mask, mask)

    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0


def test_image_labels_include_normal_and_largest_detection():
    assert image_label_from_class_ids([]) == 2
    detections = [
        Detection(
            class_id=0,
            class_name="granuloma",
            confidence=0.8,
            bbox=Box(x=0, y=0, width=5, height=5),
            polygon=[],
            mask_area=10,
        ),
        Detection(
            class_id=1,
            class_name="kista",
            confidence=0.9,
            bbox=Box(x=0, y=0, width=7, height=7),
            polygon=[],
            mask_area=30,
        ),
    ]

    assert image_label_from_detections(detections) == 1
