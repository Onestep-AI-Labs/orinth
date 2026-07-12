# NLP Capabilities

While Onestep AI Platform is deeply rooted in Image Intelligence, Phase 6 extended its scope to support critical Natural Language Processing (NLP) workflows, turning the application into a versatile workspace.

![Dataset Studio NLP Classification](/brand/5_dataset_studio_nlp_clasification_task.png)

## Supported NLP Tasks

- **Text Classification**
- **Text Summarization**
- **Question Answering**

## NLP Dataset Studio

- **Formats:** Import raw text via `.txt` folders, or structured text via `.jsonl` and `.csv` files.
- **Annotations:** The Annotation Studio automatically shifts mode for NLP tasks:
  - *Classification:* Assign categories to text.
  - *Summarization:* Edit reference summary texts.
  - *Question Answering:* Input and edit question/answer pairs.
- **Preprocessing:** Apply lowercase filtering, remove punctuation, and strip stop-words.
- **Augmentation:** Enhance datasets with synonym replacement, word swapping, and targeted word deletions.

## NLP Models & Baselines

- **Offline Baselines:** Perform instant local smoke-tests using `keyword_text_classifier`, `extractive_summarizer`, and `keyword_qa` models that run without requiring heavy transformer downloads.
- **Neural & Transformers:** Run full training jobs using Keras models (CNN, LSTM, BiLSTM, seq2seq) or Hugging Face Transformers (BERT classification/QA, BART summarization). The platform integrates natively with Hugging Face Hub (via tokens) to download checkpoints and begin task-head fine-tuning dynamically.
