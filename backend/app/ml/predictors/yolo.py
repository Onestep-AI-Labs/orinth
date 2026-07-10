from pathlib import Path

from app.ml.predictors.base import Predictor
from app.schemas import Box, Detection, InferenceParameters


class YoloPredictor(Predictor):
    def __init__(self, weights_path: Path, labels: list[str]) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.class_names = {index: label for index, label in enumerate(labels)}

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        import cv2
        import numpy as np

        results = self.model(
            str(image_path),
            verbose=False,
            conf=parameters.confidence_threshold,
            iou=parameters.iou_threshold,
        )
        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        masks = result.masks.xy if result.masks is not None else []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for index, (box, cls_id_raw, conf_raw) in enumerate(zip(boxes, classes, confidences)):
            cls_id = int(cls_id_raw)
            conf = float(conf_raw)
            x1, y1, x2, y2 = [float(value) for value in box]
            if index < len(masks):
                polygon_np = np.asarray(masks[index], dtype=float)
            else:
                polygon_np = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)

            polygon = polygon_np.tolist()
            area = float(abs(cv2.contourArea(polygon_np.astype("float32"))))
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                    confidence=conf,
                    bbox=Box(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1)),
                    polygon=polygon,
                    mask_area=area,
                )
            )
        return detections
