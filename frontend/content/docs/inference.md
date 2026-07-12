# Inference

Serve local inference for registered model families effortlessly. The Inference UI connects to all registered models from your Model Zoo, providing an easy-to-use form to test individual images or text queries.

![Inference Image Segmentation](/brand/7_inference_image_segmentation_task.png)

## Predictors & Execution

- **Lazy Loaded Models:** Predictors load lazily, ensuring memory is optimized until a prediction is explicitly requested.
- **Task-Aware Settings:** Inference parameters adapt based on the selected model. Classification models hide threshold sliders, whereas detection and segmentation tasks expose IoU and Confidence thresholds.
- **Inline Results:** Get normalized bounding boxes, polygons, and labels back instantly.

![Inference Image Classification](/brand/8_inference_image_clasification_task.png)

## History and Logging

Every prediction is logged in the project's history.
- Historical prediction payloads, alongside execution metrics like `duration_ms` and step `timings`, are stored and reviewable.
- You can filter the history by project or clear it to purge database rows and related visual overlay artifacts securely.
