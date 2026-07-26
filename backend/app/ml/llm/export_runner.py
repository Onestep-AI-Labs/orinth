"""Subprocess entrypoints for LLM export steps (phase 15).

Invoked as ``python -m app.ml.llm.export_runner <merge|quantize> ...`` by the
export service. Each step runs out-of-process for memory isolation — a 4B
merge peaks well above the API process budget — and so a native crash in
llama.cpp quantization can never take the API down with it.

The HF→GGUF f16 conversion step is not here: the service runs the pinned
``convert_hf_to_gguf.py`` (see ``app.ml.llm.gguf_tools``) directly as its own
subprocess.
"""

import argparse
import os
from pathlib import Path
from typing import Any


def _hf_env_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = (os.environ.get(key) or "").strip()
        if token:
            return token
    return None


def run_merge(adapter_dir: Path, base_ref: str, out_dir: Path, cache_dir: Path | None) -> None:
    """PEFT ``merge_and_unload`` on CPU, saved as fp16 safetensors + tokenizer.

    CPU keeps the peak footprint in ordinary RAM instead of a small GPU/MPS
    budget; merging is a one-shot weight walk, so speed matters less than not
    dying on load.
    """
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    common_kwargs = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        common_kwargs["cache_dir"] = str(cache_dir)
    token = _hf_env_token()
    if token:
        common_kwargs["token"] = token

    print(f"loading base model: {base_ref}")
    model: Any = AutoModelForCausalLM.from_pretrained(base_ref, dtype=torch.float16, **common_kwargs)
    print(f"attaching adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("merging adapter into base (merge_and_unload)")
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    # The adapter dir carries the tokenizer the run trained with (including any
    # applied chat template); fall back to the base tokenizer for adapters
    # uploaded without one.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(base_ref, **common_kwargs)
    tokenizer.save_pretrained(str(out_dir))
    # A fast tokenizer's save drops tokenizer.model; without it the GGUF
    # converter falls back to the gpt2 vocab path and aborts on Gemma. Restore
    # the SentencePiece vocab from the adapter dir or the base model.
    from app.ml.llm.gguf_tools import ensure_sentencepiece_vocab  # noqa: PLC0415

    token = _hf_env_token()
    copied = ensure_sentencepiece_vocab(out_dir, str(adapter_dir), token=token)
    if not copied:
        ensure_sentencepiece_vocab(
            out_dir, base_ref, cache_dir=common_kwargs.get("cache_dir"), token=token
        )
    print(f"merged model saved to: {out_dir}")


# llama.cpp ftype ids, mirrored from llama.h; getattr against the installed
# wheel wins when available so a renumbering upstream cannot silently corrupt.
_QUANT_FTYPES = {
    "f16": ("LLAMA_FTYPE_MOSTLY_F16", 1),
    "q8_0": ("LLAMA_FTYPE_MOSTLY_Q8_0", 7),
    "q4_k_m": ("LLAMA_FTYPE_MOSTLY_Q4_K_M", 15),
    "q5_k_m": ("LLAMA_FTYPE_MOSTLY_Q5_K_M", 17),
}


def run_quantize(input_path: Path, output_path: Path, quant_type: str) -> None:
    """Quantize a GGUF via the llama-cpp-python prebuilt wheel's C API."""
    import ctypes  # noqa: PLC0415

    import llama_cpp  # noqa: PLC0415

    constant_name, fallback = _QUANT_FTYPES[quant_type]
    ftype = int(getattr(llama_cpp, constant_name, fallback))
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = ftype
    print(f"quantizing {input_path.name} -> {output_path.name} ({quant_type})")
    result = llama_cpp.llama_model_quantize(
        str(input_path).encode("utf-8"),
        str(output_path).encode("utf-8"),
        ctypes.byref(params),
    )
    if result != 0:
        raise SystemExit(f"llama.cpp quantization failed with code {result}")
    print(f"quantized model saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--adapter-dir", required=True)
    merge.add_argument("--base-ref", required=True)
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--hf-cache-dir", default="")

    quantize = subparsers.add_parser("quantize")
    quantize.add_argument("--input", required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--type", required=True, choices=sorted(_QUANT_FTYPES))

    args = parser.parse_args()
    if args.command == "merge":
        run_merge(
            Path(args.adapter_dir),
            args.base_ref,
            Path(args.out_dir),
            Path(args.hf_cache_dir) if args.hf_cache_dir else None,
        )
    else:
        run_quantize(Path(args.input), Path(args.output), args.type)


if __name__ == "__main__":
    main()
