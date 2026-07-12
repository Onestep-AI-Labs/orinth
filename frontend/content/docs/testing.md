# Testing & Evaluation

Run local evaluation jobs against your image and text dataset splits and immediately store resulting metrics and curve artifacts for comparison.

## Evaluation Jobs

- **Comparison Runs:** Select multiple models across the same dataset to create a batch comparison run. Testing will group these into a single `comparison_id`, letting you stack models side-by-side to understand performance trade-offs.
- **Task-Aware Metrics:** The UI automatically chooses which metrics to surface based on the task:
  - *Classification:* Emphasizes Accuracy, F1, MCC, and AUC, paired with a Confusion Matrix.
  - *Detection/Segmentation:* Emphasizes Pixel Dice, object IoU matching, Precision, and Recall.
  - *NLP:* Exact Match, ROUGE scores, etc.

## Per-Item Inspection

Stop staring at aggregate numbers and investigate where your models fail. The Per-Item inspection table renders predictions row by row:
- View ground-truth classes versus predicted classes with raw accuracy scores.
- For vision, view precise label overlap, pixel metrics, and object matching summaries for each individual image test.
- For NLP, easily compare original texts, reference answers, and model predictions directly in the table.
