# LLM Fine-tuning & Chat

Onestep AI Platform fine-tunes language models, serves the result locally, and lets you chat with it — all inside the same project workspace. This is a research instrument for measuring model behavior, not a medical or clinical assistant.

## Choosing a base model

LLM fine-tuning runs under the `llm_finetune` task on the **Training** page. The base model is a Hugging Face Transformers checkpoint (safetensors/PyTorch with `config.json` and a tokenizer), taken either from the Hub or from a local `llm_hf` upload.

- Gated Hub models require accepting the license on huggingface.co and a **Hugging Face token** set under **Settings**.
- Live model details — download size, file count, and gating — are read from the Hub as you select a base.
- GGUF and TFLite are inference/export formats and cannot be used as a training base.

## Fine-tuning methods

| Method | What it does |
| --- | --- |
| **LoRA adapter** | Trains a small tuned delta over the base. Light and recommended. |
| **QLoRA (4-bit)** | Lowest VRAM; needs CUDA. On MPS/CPU it trains as plain LoRA. |
| **Full fine-tune** | Updates every weight. Heavy — prefer LoRA/QLoRA on a Mac or CPU. |
| **Continued pretraining** | LoRA over full text for domain adaptation. |

The platform detects your accelerator (CUDA, Apple MPS, or CPU) and picks a matching backend, adjusting defaults such as 4-bit and precision so the shown configuration matches what the run will actually do.

## Running and observing

LLM runs execute as subprocesses and open the live training detail view on start, streaming progress and logs while they run.

![LLM fine-tuning run detail](/brand/11_llm_training_detail.png)

## Exporting and serving

A finished LLM run produces one of three artifact families, each with a plain-language note in the catalog:

- **`llm_adapter`** — a LoRA adapter (the tuned delta, not a standalone model). Export to GGUF to serve and chat, or to merged 16-bit to download a full model.
- **`llm_hf`** — a full Hugging Face checkpoint. Export to GGUF to serve it in the chat runtime.
- **`llm_gguf`** — a quantized, self-contained model. Serve it and chat, or re-quantize to a smaller size.

Only GGUF models are servable, so export first if your artifact is an adapter or a full checkpoint. Export and re-quantization actions live on the model detail page — see **Model Catalog & Serving**.

## Chatting with a served model

Open **Inference → Chat** (or the chat action on a GGUF model card) to start serving and hold a conversation.

![Chat with a served GGUF model](/brand/12_llm_chat.png)

- **Served model** — start, stop, or switch the GGUF model from the left rail. Serving brings the runtime up and reports uptime.
- **Sampler** — temperature, top-p, and max tokens shape each reply. A system prompt sets behavior for the whole conversation.
- **Web search** — toggle search to fetch results, cite them inline as `[n]`, and ground the answer in sources.
- **Thinking** — show or hide the model's reasoning for models that emit it.
- **Throughput** — completion tokens, tokens/second, and time-to-first-token are reported under the transcript.
