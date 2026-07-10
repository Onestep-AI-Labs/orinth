IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("unassigned", "train", "valid", "test")
TRAINING_SPLITS = ("train", "valid", "test")
IMAGE_TASK_TYPES = {"classification", "object_detection", "segmentation"}

ALLOWED_PREPROCESS_TRANSFORMS = {
    "horizontal_flip",
    "vertical_flip",
    "brightness_contrast",
    "gaussian_blur",
}

PREPROCESS_PRESETS = {
    "none": [],
    "light": ["horizontal_flip", "brightness_contrast"],
    "inspection": ["brightness_contrast"],
}
