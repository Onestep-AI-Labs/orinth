"""Constants for phase-11 data recipes."""

# Upload acceptance. Kept small and pure-Python: pypdf/python-docx for PDF/DOCX,
# stdlib for the rest. No heavy ML libraries are ever imported in this path.
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".csv", ".jsonl"}
MAX_SOURCE_BYTES = 20 * 1024 * 1024  # 20 MB per file.

RECIPE_STATUSES = ("draft", "extracting", "generating", "ready", "failed")
# Statuses that cannot survive a backend restart (executor work is in-memory);
# the startup sweep flips any manifest left here to `failed`.
TRANSIENT_STATUSES = ("extracting", "generating")

OUTPUT_FORMATS = ("instruction_jsonl", "chat_jsonl")
PROMPT_FLAVORS = ("qa", "instruction", "conversation")

DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RECORDS_PER_CHUNK = 3

# A small curated fallback so the model select never renders empty when no key
# is configured or the OpenRouter `/models` call fails.
FALLBACK_OPENROUTER_MODELS = [
    ("openai/gpt-4o-mini", "GPT-4o mini"),
    ("anthropic/claude-3.5-haiku", "Claude 3.5 Haiku"),
    ("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B Instruct"),
    ("google/gemini-flash-1.5", "Gemini Flash 1.5"),
    ("mistralai/mistral-7b-instruct", "Mistral 7B Instruct"),
]
