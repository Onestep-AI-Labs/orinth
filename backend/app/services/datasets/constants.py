IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_SUFFIXES = {".txt", ".csv", ".jsonl"}
RECORD_SUFFIX = ".json"
SPLITS = ("unassigned", "train", "valid", "test")
TRAINING_SPLITS = ("train", "valid", "test")
IMAGE_TASK_TYPES = {"classification", "object_detection", "segmentation"}
NLP_TASK_TYPES = {"text_classification", "summarization", "question_answering"}
# LLM fine-tuning: one task, two record shapes selected by format (see phase 10).
LLM_TASK_TYPES = {"llm_finetune"}
LLM_FORMATS = {"instruction_jsonl", "chat_jsonl"}
CHAT_ROLES = ("system", "user", "assistant")
TASK_TYPE_ALIASES = {"text": "text_classification"}
# Dataset provenance, stored on the manifest (see phase 10). Defaulted to
# ``created`` for legacy manifests written before this field existed.
DATASET_ORIGINS = ("created", "imported_hf", "recipe")

ALLOWED_PREPROCESS_TRANSFORMS = {
    "horizontal_flip",
    "vertical_flip",
    "brightness_contrast",
    "gaussian_blur",
    "lowercase",
    "remove_punctuation",
    "remove_stopwords",
    "normalize_whitespace",
    "synonym_replacement",
    "word_swap",
    "word_deletion",
}

PREPROCESS_PRESETS = {
    "none": [],
    "light": ["horizontal_flip", "brightness_contrast"],
    "inspection": ["brightness_contrast"],
    "nlp_clean": ["lowercase", "remove_punctuation", "normalize_whitespace"],
    "nlp_augment": ["synonym_replacement", "word_swap", "word_deletion"],
}
