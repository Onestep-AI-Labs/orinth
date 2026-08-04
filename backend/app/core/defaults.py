DEFAULT_PROJECT_ID = "default-research-project"
DEFAULT_PROJECT_NAME = "Research workspace"
# Name this project carried before it covered every task type. Retained so the
# migration that renames it can tell an untouched default from one a user
# deliberately renamed.
LEGACY_DEFAULT_PROJECT_NAME = "Research Image Workspace"
DEFAULT_TASK_TYPE = "segmentation"
# The starter project is not vision-only: it enables every task type so a fresh
# install can go straight to any workflow without editing project settings.
# Keep in sync with `TaskType` in app/schemas.py.
DEFAULT_PROJECT_TASK_TYPES = [
    "classification",
    "object_detection",
    "segmentation",
    "text_classification",
    "summarization",
    "question_answering",
    "llm_finetune",
    "language_modeling",
]
DEFAULT_LABELS = ["granuloma", "kista"]
NORMAL_LABEL = "Normal"
