# Spec: Phase 6 NLP Tasks

## Status

Implemented V1 with offline local baselines

## Goal

Extend the Onestep AI Platform workspace to support Natural Language Processing (NLP) tasks:
- Text Classification
- Text Summarization
- Question Answering

This extension enables users to create projects with NLP task types, upload/import text data, clean and augment text, train simple NLP models, evaluate model predictions using standard NLP metrics, and run text-based inference.

## Scope

### In:
- **Project & Task Types**: Add `text_classification`, `summarization`, and `question_answering` to allowlisted project task types.
- **Dataset Studio**:
  - Support NLP dataset formats: `text_folder`, `jsonl`, and `csv`.
  - Ship tracked sample NLP datasets with at least 20 annotated text items per NLP task.
  - Allow uploading/importing `.txt` files or importing structured `.jsonl` / `.csv` data.
  - Provide an Annotation Studio for NLP:
    - `text_classification`: Select category label.
    - `summarization`: Input and edit a reference summary text.
    - `question_answering`: Input and edit question-answer pairs.
  - Add text preprocessing options: Lowercase, remove punctuation, and remove stop words.
  - Add text augmentation options: Synonym replacement, word swap, and word deletion.
  - Support generating versioned/augmented text datasets under `storage/dataset_versions`.
- **Model Registry & Predictors**:
  - Add reference model specifications for offline NLP baselines:
    - `keyword_text_classifier` (Text Classification, runnable)
    - `extractive_summarizer` (Summarization, runnable)
    - `keyword_qa` (Question Answering, runnable)
  - Implement predictors for NLP tasks that read input text and parameters, executing predictions without transformer downloads.
- **Training**:
  - Add training options for NLP task types.
  - Implement a runnable `app/training/runners/nlp_train.py` script that trains TF-IDF + Logistic Regression for text classification and lightweight extractive JSON baselines for summarization / QA, generating `results.csv`, `metrics.json`, `validation_predictions.json`, and model artifacts.
- **Testing & Evaluation**:
  - Support testing comparison runs on NLP datasets.
  - Implement NLP evaluation metrics:
    - `text_classification`: Accuracy, F1-score, Precision, Recall, Confusion Matrix.
    - `summarization`: ROUGE-1, ROUGE-2, ROUGE-L.
    - `question_answering`: Exact Match (EM) and F1-score.
- **Inference**:
  - Expose text inference through `InferenceService`, supporting question-context query parameters.

### Out:
- Pre-training massive LLMs from scratch.
- Complex multi-lingual translations (limited to English-based tokenization/cleaning/stopwords).

## Interfaces

### API Changes
- No new routes are needed. Existing routes are generalized to accept:
  - `TaskType`: `"text_classification"`, `"summarization"`, `"question_answering"`.
  - `DatasetFormat`: `"text_folder"`, `"jsonl"`, `"csv"`.
- Modify `DatasetAnnotation` schema to add optional NLP fields:
  - `text: str | None = None` (used for summarization reference text or QA answer)
  - `question: str | None = None` (used for QA question)
  - `answer: str | None = None` (used for QA answer)
- Modify `InferenceResult` schema to include:
  - `text_content: str | None = None`
  - `nlp_result: dict[str, Any] | None = None`
- Modify `InferenceParameters` to add:
  - `question: str | None = None`
  - `max_length: int = 120`

### V1 Base Model Policy
- V1 is intentionally offline-first. Hugging Face BART/DistilBERT-style transformer downloads are out of scope for the first runnable implementation.
- Future transformer entries may be added as gated catalog items after dependency and asset preparation flows are validated.

### Storage & DB
- Existing database tables `projects`, `inference_runs`, `inference_jobs`, `evaluation_jobs`, and `training_jobs` utilize flexible `JSON` columns (`metadata`, `result`, `parameters`, `metrics`, `artifacts`) and thus do not require database migration.
- NLP dataset files are stored under `storage/datasets/{dataset_id}/{split}/texts/{item_id}.txt` instead of `images/`. Annotations are stored under `storage/datasets/{dataset_id}/{split}/annotations/{item_id}.json`.

## Data Flow

### Upload/Import
1. User uploads a `.txt` file -> stored under `texts/` folder.
2. User imports a `.jsonl` or `.csv` file -> parser splits it into individual `.txt` items and matching `.json` annotation files.

### Annotation Editing
1. User saves label or inputs QA/summary in the UI -> backend writes to `annotations/{item_id}.json`.

### Preprocessing & Augmentation
1. When generating a version, the backend cleans text (lowercase, punctuation/stopword removal) or augments it (synonym swap, word insertion/deletion) and saves the processed text files under `storage/dataset_versions`.

### Training
1. Frontend calls `POST /api/training/jobs`.
2. Backend spawns `app/training/runners/nlp_train.py`.
3. Runner computes TF-IDF + Classifier or mimics Transformer training, outputs `metrics.json` and `validation_predictions.json`.
4. Backend promotes/registers the model weights.

### Testing
1. Frontend calls `POST /api/testing/jobs`.
2. Backend computes predictions on the test split, scores metrics, and writes the output.

## Acceptance Criteria
- User can create a project with NLP task types (`text_classification`, `summarization`, `question_answering`).
- User can create a dataset with NLP format, upload text files, edit annotations, and run cleaning/augmentations.
- Each bundled NLP task sample dataset has at least 20 annotated text items across train, valid, and test splits.
- User can train NLP models, monitor training logs, view evaluation comparison results, and run text-based inference.
