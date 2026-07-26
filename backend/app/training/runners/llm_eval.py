"""Subprocess entrypoint for LLM held-out evaluation (phase 15).

Invoked as ``python -m app.training.runners.llm_eval``. Loads a fine-tuned
model (base + optional PEFT adapter, or a merged HF checkpoint) and scores it
on a dataset split with the standard held-out metrics: perplexity, mean
cross-entropy loss, and next-token accuracy over the assistant tokens only.

Runs out of process for memory isolation — the same reason SFT does — and
reuses the SFT data pipeline (chat templating + completion masking) so the
tokens scored here are exactly the tokens the model was trained to predict.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from app.training.runners.llm_sft import (
    _select_device,
    ensure_chat_template,
    load_jsonl_records,
    prepare_examples,
    record_messages,
)

# Below this the perplexity estimate is too noisy to be meaningful; the eval
# still runs but the service surfaces the count so the number is read in context.
MIN_EVAL_RECORDS = 1
PROGRESS_EVERY = 5


def load_model_for_eval(model_ref: str, adapter_dir: Path | None, device, cache_dir: Path | None):
    """Load base (+ optional adapter) or a merged HF model in eval mode."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    common: dict[str, Any] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        common["cache_dir"] = str(cache_dir)

    # The adapter dir carries the tokenizer the run trained with (chat template
    # included); fall back to the base model's tokenizer otherwise.
    tokenizer_source = str(adapter_dir) if adapter_dir is not None else model_ref
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **common)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_ref, **common)

    dtype = torch.float32
    if device.type in {"cuda", "mps"}:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_ref, dtype=dtype, **common)
    if adapter_dir is not None:
        from peft import PeftModel  # noqa: PLC0415

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.eval()
    return model, tokenizer


def score_example(model, example: dict, device) -> tuple[float, int, int]:
    """Return (summed loss, scored-token count, correct next-token count) for one example.

    Loss is summed (not averaged) so the corpus perplexity weights every scored
    token equally regardless of how examples are batched.
    """
    import torch  # noqa: PLC0415

    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([example["attention_mask"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift: predict token t+1 from position t.
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    scored = int(mask.sum().item())
    if scored == 0:
        return 0.0, 0, 0
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)
    losses = torch.nn.functional.cross_entropy(
        flat_logits, flat_labels.clamp(min=0), reduction="none"
    )
    valid = mask.reshape(-1)
    summed_loss = float((losses * valid).sum().item())
    predictions = flat_logits.argmax(dim=-1)
    correct = int(((predictions == flat_labels) & valid).sum().item())
    return summed_loss, scored, correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    records = load_jsonl_records(dataset_root, args.split)
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    if len(records) < MIN_EVAL_RECORDS:
        raise SystemExit(
            f"No evaluation records found in the '{args.split}' split of {dataset_root}. "
            "Assign records to a test split before evaluating."
        )
    print(f"eval records={len(records)} split={args.split}")

    device = _select_device()
    print(f"evaluating on device: {device}")
    print(f"Loading model: {args.model_ref}" + (f" + adapter {args.adapter_dir}" if args.adapter_dir else ""))
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    model, tokenizer = load_model_for_eval(
        args.model_ref, adapter_dir, device, Path(args.hf_cache_dir) if args.hf_cache_dir else None
    )
    ensure_chat_template(tokenizer)

    examples, stats = prepare_examples(
        tokenizer, records, max_seq_length=args.max_seq_length, mask_prompt=True
    )
    if not examples:
        raise SystemExit(
            "No scorable examples remained after templating/masking. Check the split records."
        )
    if stats["skipped"]:
        print(f"skipped {stats['skipped']} malformed or fully-masked records")

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    per_record: list[dict[str, Any]] = []
    total = len(examples)
    for index, example in enumerate(examples, start=1):
        summed_loss, scored, correct = score_example(model, example, device)
        total_loss += summed_loss
        total_tokens += scored
        total_correct += correct
        record = records[index - 1] if index - 1 < len(records) else {}
        messages = record_messages(record) or []
        prompt_preview = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        reference = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"), ""
        )
        record_loss = summed_loss / scored if scored else 0.0
        per_record.append(
            {
                "image": f"record {index}",
                "ground_truth": 0,
                "prediction": 0,
                "detections": 0,
                "objects": 1,
                "pixel": {},
                "object": {},
                "text_preview": str(prompt_preview)[:180],
                "reference_text": str(reference)[:400],
                "prediction_text": "",
                "scores": {
                    "loss": round(record_loss, 6),
                    "perplexity": round(math.exp(min(record_loss, 20.0)), 4),
                    "scored_tokens": scored,
                },
            }
        )
        if index == 1 or index == total or index % PROGRESS_EVERY == 0:
            print(f"eval {index}/{total}")

    mean_loss = total_loss / total_tokens if total_tokens else 0.0
    metrics = {
        "samples": len(examples),
        "labels": [],
        "llm": {
            "loss": round(mean_loss, 6),
            "perplexity": round(math.exp(min(mean_loss, 20.0)), 4),
            "token_accuracy": round(total_correct / total_tokens, 6) if total_tokens else 0.0,
            "scored_tokens": total_tokens,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "per_record.json").write_text(
        json.dumps(per_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "eval complete: "
        f"perplexity={metrics['llm']['perplexity']} "
        f"loss={metrics['llm']['loss']} "
        f"token_accuracy={metrics['llm']['token_accuracy']}"
    )


if __name__ == "__main__":
    main()
