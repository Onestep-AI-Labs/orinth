import numpy as np
from PIL import Image, ImageDraw

from app.schemas import Detection
from app.services.evaluation.types import GroundTruthObject


def polygon_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def rasterize_ground_truth(objects: list[GroundTruthObject], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for obj in objects:
        if len(obj.polygon) >= 3:
            draw.polygon([(x, y) for x, y in obj.polygon], fill=1)
    return np.asarray(mask, dtype=np.uint8)


def rasterize_detections(detections: list[Detection], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for detection in detections:
        if len(detection.polygon) >= 3:
            draw.polygon([(x, y) for x, y in detection.polygon], fill=1)
    return np.asarray(mask, dtype=np.uint8)


def classification_roc_auc(y_true: list[int], score_rows: list[list[float]], labels: list[str]) -> dict:
    if not y_true or not score_rows or len(labels) < 2:
        return {"roc_curves": {}, "macro_auc": None, "micro_auc": None}

    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    scores = np.asarray(score_rows, dtype=float)
    label_indices = list(range(len(labels)))
    y_bin = label_binarize(y_true, classes=label_indices)
    if len(labels) == 2 and y_bin.ndim == 2 and y_bin.shape[1] == 1:
        y_bin = np.concatenate([1 - y_bin, y_bin], axis=1)

    result: dict = {"roc_curves": {}, "macro_auc": None, "micro_auc": None}
    try:
        result["macro_auc"] = round(
            float(roc_auc_score(y_bin, scores, average="macro", multi_class="ovr")),
            6,
        )
        result["micro_auc"] = round(
            float(roc_auc_score(y_bin, scores, average="micro", multi_class="ovr")),
            6,
        )
    except ValueError:
        pass

    for index, label in enumerate(labels):
        try:
            fpr, tpr, thresholds = roc_curve(y_bin[:, index], scores[:, index])
        except ValueError:
            continue
        result["roc_curves"][label] = {
            "fpr": [round(float(value), 6) for value in fpr],
            "tpr": [round(float(value), 6) for value in tpr],
            "thresholds": [round(float(value), 6) for value in thresholds],
        }
    return result


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: round(float(np.mean([row[key] for row in rows])), 6) for key in keys}
