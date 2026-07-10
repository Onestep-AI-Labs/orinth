from collections import Counter
from typing import Iterable

import numpy as np

from app.core.defaults import DEFAULT_LABELS, NORMAL_LABEL
from app.schemas import Detection


def binary_mask_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = y_true.astype(bool).reshape(-1)
    pred = y_pred.astype(bool).reshape(-1)
    tp = float(np.logical_and(true, pred).sum())
    tn = float(np.logical_and(~true, ~pred).sum())
    fp = float(np.logical_and(~true, pred).sum())
    fn = float(np.logical_and(true, ~pred).sum())

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)

    return {
        "dice": round(dice, 6),
        "iou": round(iou, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
    }


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-7)


def image_label_from_class_ids(class_ids: Iterable[int], normal_index: int | None = None) -> int:
    counts = Counter(class_ids)
    normal = normal_index if normal_index is not None else len(DEFAULT_LABELS)
    if not counts:
        return normal
    return counts.most_common(1)[0][0]


def image_label_from_detections(
    detections: list[Detection], normal_index: int | None = None
) -> int:
    normal = normal_index if normal_index is not None else len(DEFAULT_LABELS)
    if not detections:
        return normal
    largest = max(detections, key=lambda item: item.mask_area)
    return largest.class_id


def classification_metrics(y_true: list[int], y_pred: list[int], class_names: list[str] | None = None) -> dict:
    if not y_true:
        return {}

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
    )

    names = class_names or DEFAULT_LABELS
    label_names = {index: name for index, name in enumerate(names)}
    label_names[len(names)] = NORMAL_LABEL
    labels = sorted(set(y_true + y_pred))
    target_names = [label_names.get(label, str(label)) for label in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    for idx, label in enumerate(labels):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        tn = float(cm.sum() - (tp + fp + fn))
        sensitivity = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        lr_plus = sensitivity / (1.0 - specificity + 1e-7)
        lr_minus = (1.0 - sensitivity) / (specificity + 1e-7)
        per_class.append(
            {
                "label": label,
                "class_name": label_names.get(label, str(label)),
                "sensitivity": round(sensitivity, 6),
                "specificity": round(specificity, 6),
                "lr_plus": round(lr_plus, 6),
                "lr_minus": round(lr_minus, 6),
            }
        )

    return {
        "labels": target_names,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "overall": {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
            "weighted_f1": round(
                float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6
            ),
            "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 6),
            "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
        },
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }
