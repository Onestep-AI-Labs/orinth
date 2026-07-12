IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_SUFFIXES = {".txt", ".csv", ".jsonl"}
SPLITS = ("unassigned", "train", "valid", "test")
TRAINING_SPLITS = ("train", "valid", "test")
IMAGE_TASK_TYPES = {"classification", "object_detection", "segmentation"}
NLP_TASK_TYPES = {"text_classification", "summarization", "question_answering"}
TASK_TYPE_ALIASES = {"text": "text_classification"}

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
