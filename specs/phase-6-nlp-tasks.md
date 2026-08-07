# Spec: Phase 6 NLP Tasks

## Status

Implemented V1 with local baselines; neural Keras and Hugging Face transformer adoption in progress

## Goal

Extend the workspace to support Natural Language Processing (NLP) tasks:
- Text Classification
- Text Summarization
- Question Answering

This extension enables users to create projects with NLP task types, upload/import text data, clean and augment text, train simple NLP models, evaluate model predictions using standard NLP metrics, and run text-based inference.

## Scope

### In:
- **Project & Task Types**: Add `text_classification`, `summarization`, and `question_answering` to allowlisted project task types.
- **Dataset Studio**:
  - Support NLP dataset formats: `text_folder`, `jsonl`, and `csv`.
  - Ship tracked sample NLP datasets with at least 100 annotated text items per NLP task.
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
  - Organize ML code by task and model family under `backend/app/ml/nlp/{model_family}`.
  - Add trained-model predictors for Keras text classifiers, Keras seq2seq summarizers, and Hugging Face transformer artifacts.
- **Training**:
  - Add training options for NLP task types.
  - Implement a runnable `app/training/runners/nlp_train.py` script that trains TF-IDF + Logistic Regression for text classification and lightweight extractive JSON baselines for summarization / QA, generating `results.csv`, `metrics.json`, `validation_predictions.json`, and model artifacts.
  - Add runnable Keras text-classification options: CNN, LSTM, and BiLSTM.
  - Add runnable Keras seq2seq summarization.
  - Add runnable Hugging Face options for BERT text classification, BERT extractive QA, and BART summarization.
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
- V1 keeps offline baselines available for fast local smoke tests.
- Neural NLP training is task-first and catalog-driven.
- Hugging Face transformer options are runnable when dependencies and model assets are prepared locally. Private/gated assets use `HUGGINGFACE_HUB_TOKEN`.
- The backend also accepts `HF_TOKEN` as an alias for Hugging Face access, and the Settings page persists `HUGGINGFACE_HUB_TOKEN` to the ignored workspace `.env` without returning the secret value.
- The BERT family is offered as several checkpoints sharing one runner and predictor per task, since both build through `AutoTokenizer` / `AutoModelFor*`. Classification offers DistilBERT, RoBERTa, BERT (uncased and cased), ALBERT, and multilingual BERT; question answering offers SQuAD-pretrained DistilBERT and RoBERTa alongside plain BERT and multilingual BERT.
- Each task lists its best small-data default first, because the training page auto-selects the first runnable option.
- Classification treats task-head initialization as expected fine-tuning behavior. Extractive QA does not: a checkpoint already fine-tuned on SQuAD arrives with a trained span head and is the recommended default, because a freshly initialized span head needs far more than 100 examples to become useful.
- Transformer fine-tuning follows the standard recipe — shuffled batches, linear warmup and decay, weight decay excluding bias and LayerNorm, gradient clipping, per-epoch validation, and best-checkpoint selection. On a corpus this small, omitting these collapses the model to a constant prediction.
- Validation is never the training split. When a `valid` split is missing or smaller than five items, a stratified holdout is carved from train and the metrics record `validation_source`.
- QA training examples whose answer is not a literal span of their context are skipped and counted, never supervised at position 0 — which for `[CLS] question [SEP] context` teaches the model to echo the question.
- Training runs on CUDA or MPS when available, falling back to CPU; `ONESTEP_TRAIN_DEVICE` overrides the choice.

### Storage & DB
- Existing database tables `projects`, `inference_runs`, `inference_jobs`, `evaluation_jobs`, and `training_jobs` utilize flexible `JSON` columns (`metadata`, `result`, `parameters`, `metrics`, `artifacts`) and thus do not require database migration.
- NLP dataset files are stored under `storage/datasets/{dataset_id}/{split}/texts/{item_id}.txt` instead of `images/`. Annotations are stored under `storage/datasets/{dataset_id}/{split}/annotations/{item_id}.json`.
- Bundled sample NLP datasets are visible in project-scoped dataset selectors, including the default workspace, and split summaries report text counts separately from image counts.

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
3. Runner computes the selected local baseline, Keras model, or Hugging Face transformer training path, outputs `metrics.json` and `validation_predictions.json`.
4. Backend promotes/registers the model weights.

### Testing
1. Frontend calls `POST /api/testing/jobs`.
2. Backend computes predictions on the test split, scores metrics, and writes the output.

## Acceptance Criteria
- User can create a project with NLP task types (`text_classification`, `summarization`, `question_answering`).
- The default research workspace exposes NLP task types and bundled sample NLP datasets.
- User can create a dataset with NLP format, upload text files, edit annotations, and run cleaning/augmentations.
- Each bundled NLP task sample dataset has at least 100 annotated text items across train, valid, and test splits, with a `valid` split of at least 15.
- User can train NLP models, monitor training logs, view evaluation comparison results, and run text-based inference.
- Text classification model options include TF-IDF, Keras CNN, Keras LSTM, Keras BiLSTM, and the Hugging Face BERT family (DistilBERT, RoBERTa, BERT, BERT cased, ALBERT, multilingual BERT).
- Summarization model options include extractive baseline, Keras seq2seq, and Hugging Face BART.
- Question answering model options include keyword QA, SQuAD-pretrained DistilBERT and RoBERTa, plain BERT QA, and multilingual BERT QA.
- `results.csv` for a Hugging Face run carries a per-epoch validation curve, not training loss alone, so overfitting is visible.
- Fine-tuning the bundled 100-item text classification sample reaches a macro-F1 of at least 0.70 on its own valid split, with every class predicted at least once.
