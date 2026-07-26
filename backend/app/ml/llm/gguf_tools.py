"""Pinned llama.cpp HF→GGUF converter tooling (phase 15).

``convert_hf_to_gguf.py`` is fetched on first use from the pinned llama.cpp
tag into ``storage/tools/llama.cpp/<tag>/`` and checksum-verified before it is
ever executed. Rationale: pip-only install for everything else
(``llama-cpp-python`` + ``gguf`` ride the ``llm`` extra), no user-side
compilation, and no AGPL code imported into the app — the script runs as a
separate subprocess.

The tag is pinned to the last release whose converter is a single
self-contained script (later tags split it into a repo-local ``conversion``
package that cannot be fetched as one verifiable file). b9000 supports the
Qwen3.5, Gemma 4, and Ministral 3 architectures the spec requires, alongside
the phase-14 catalog families (Qwen2.5, SmolLM2, Gemma 3).
"""

import hashlib
import shutil
import urllib.request
from pathlib import Path

# Vocab files the llama.cpp converter's SentencePiece path needs but a fast
# tokenizer's ``save_pretrained`` often omits. Copying them from the base keeps
# SentencePiece models (Gemma, Llama) on the correct conversion path.
SENTENCEPIECE_VOCAB_FILES = ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model")

LLAMA_CPP_TAG = "b9000"
CONVERTER_FILENAME = "convert_hf_to_gguf.py"
CONVERTER_URL = (
    f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_TAG}/{CONVERTER_FILENAME}"
)
CONVERTER_SHA256 = "3ff05b62f65c16c1864ee3439b692aa6f184f5e18356f94ae2ebe1c7427644d4"


class ConverterFetchError(RuntimeError):
    pass


def converter_dir(tools_root: Path) -> Path:
    return tools_root / "llama.cpp" / LLAMA_CPP_TAG


def converter_path(tools_root: Path) -> Path:
    return converter_dir(tools_root) / CONVERTER_FILENAME


def ensure_converter(tools_root: Path) -> Path:
    """Return the verified converter script path, fetching it on first use.

    A previously fetched file is re-verified on every call — a corrupted or
    tampered cache re-fetches rather than executing unverified code.
    """
    target = converter_path(tools_root)
    if target.exists() and _sha256(target) == CONVERTER_SHA256:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(CONVERTER_URL, timeout=60) as response:  # noqa: S310
            payload = response.read()
    except OSError as exc:
        raise ConverterFetchError(
            f"Could not download the GGUF converter from llama.cpp {LLAMA_CPP_TAG}: {exc}. "
            "GGUF export needs network access on first use."
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONVERTER_SHA256:
        raise ConverterFetchError(
            f"GGUF converter checksum mismatch for llama.cpp {LLAMA_CPP_TAG}: "
            f"expected {CONVERTER_SHA256}, got {digest}. Refusing to run unverified code."
        )
    tmp_path = target.with_suffix(".tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_sentencepiece_vocab(
    model_dir: Path,
    base_ref: str,
    *,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> bool:
    """Copy the base model's SentencePiece vocab into ``model_dir`` if missing.

    The llama.cpp HF→GGUF converter (see ``convert_hf_to_gguf.py``) chooses the
    SentencePiece vocab path only when ``tokenizer.model`` is present; otherwise
    it falls back to the gpt2 BPE path, which asserts on Gemma
    (``max(tokenizer.vocab.values()) < vocab_size``) and aborts the export. A
    fast tokenizer's ``save_pretrained`` writes ``tokenizer.json`` but usually
    not ``tokenizer.model``, so a merged/exported SentencePiece model loses it.
    This restores it from the base — a local dir or the Hub — so GGUF export of
    Gemma/Llama-family models succeeds. Returns True when a file was copied.

    A no-op for BPE-only models (Qwen, SmolLM2): they have no SentencePiece
    vocab and correctly use the gpt2 path.
    """
    model_dir = Path(model_dir)
    if any((model_dir / name).exists() for name in SENTENCEPIECE_VOCAB_FILES):
        return False

    base_path = Path(base_ref)
    if base_path.exists():
        for name in SENTENCEPIECE_VOCAB_FILES:
            candidate = base_path / name
            if candidate.exists():
                shutil.copy2(candidate, model_dir / name)
                return True
        return False

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    for name in SENTENCEPIECE_VOCAB_FILES:
        try:
            kwargs: dict = {}
            if cache_dir:
                kwargs["cache_dir"] = str(cache_dir)
            if token:
                kwargs["token"] = token
            fetched = hf_hub_download(base_ref, name, **kwargs)
        except Exception:  # noqa: BLE001 — model has no such vocab file; try the next
            continue
        shutil.copy2(fetched, model_dir / name)
        return True
    return False
