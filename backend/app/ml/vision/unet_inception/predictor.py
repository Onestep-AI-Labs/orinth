from pathlib import Path

from app.ml.common.base import Predictor
from app.schemas import Box, Detection, InferenceParameters


class UnetInceptionPredictor(Predictor):
    class_names = {0: "granuloma", 1: "kista"}

    def __init__(self, unet_path: Path, classifier_path: Path) -> None:
        if not unet_path.exists():
            raise FileNotFoundError(f"U-Net model not found: {unet_path}")
        if not classifier_path.exists():
            raise FileNotFoundError(f"Inception classifier not found: {classifier_path}")

        import tensorflow as tf

        def dice_coeff(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2.0 * intersection + smooth) / (
                tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
            )

        self.unet = tf.keras.models.load_model(
            unet_path,
            custom_objects={"dice_coeff": dice_coeff},
            compile=False,
        )
        self.classifier = tf.keras.models.load_model(classifier_path, compile=False)

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        import cv2
        import numpy as np
        from PIL import Image

        original = Image.open(image_path).convert("RGB")
        orig_w, orig_h = original.size
        resized = original.resize((256, 256))
        unet_input = np.expand_dims(np.asarray(resized, dtype=np.float32) / 255.0, axis=0)
        mask_probs = self.unet.predict(unet_input, verbose=0)
        pred_mask = np.argmax(mask_probs, axis=-1).squeeze().astype("uint8")

        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        scale_x = orig_w / 256.0
        scale_y = orig_h / 256.0

        for contour in contours:
            if cv2.contourArea(contour) < 8:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x1 = max(0, int(x * scale_x))
            y1 = max(0, int(y * scale_y))
            x2 = min(orig_w, int((x + w) * scale_x))
            y2 = min(orig_h, int((y + h) * scale_y))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = original.crop((x1, y1, x2, y2)).resize((299, 299))
            classifier_input = np.expand_dims(np.asarray(crop, dtype=np.float32) / 255.0, axis=0)
            scores = self.classifier.predict(classifier_input, verbose=0)[0]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < parameters.confidence_threshold:
                continue

            scaled_polygon = []
            for point in contour.reshape(-1, 2):
                scaled_polygon.append([float(point[0] * scale_x), float(point[1] * scale_y)])

            scaled_contour = np.asarray(scaled_polygon, dtype="float32")
            area = float(abs(cv2.contourArea(scaled_contour)))
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    bbox=Box(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1)),
                    polygon=scaled_polygon,
                    mask_area=area,
                )
            )

        return detections
