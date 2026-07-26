# Model Catalog & Serving

The Models page is the registry for everything you can run. It groups models by where they came from and links each one to the actions it supports — inference, testing, export, and serving.

![Model catalog grouped by source](/brand/13_model_catalog.png)

## Sources

- **Reference** — built-in baselines shipped with the platform.
- **Trained** — models produced by your training runs, registered automatically on a successful run.
- **Uploaded** — custom weights you brought in yourself.

Each card shows the task, family, size, labels, and availability, and opens a detail page for its full metadata and actions. Reference models are read-only; trained and uploaded models can be renamed, downloaded, or deleted.

## Uploading custom weights

Bring your own model into the catalog with **Upload model**. Supported artifacts include vision/NLP checkpoints, a full Hugging Face `llm_hf` checkpoint, a `llm_gguf` file, or a LoRA `llm_adapter`. Once registered, an uploaded model behaves like any other — usable for inference, testing, or (for GGUF) serving.

## Export and serving

The model detail page carries the export and serving actions that depend on the artifact family:

![Model detail — export and serve](/brand/14_model_detail_export.png)

- **Adapters and full checkpoints** export to **GGUF** (to serve) or to a **merged 16-bit** model (to download a standalone model).
- **GGUF models** can be **served and chatted with** directly, or **re-quantized** to a smaller size.

Serving and chat are covered end to end in **LLM Fine-tuning & Chat**.

## Storage

Model files managed by the registry live under local, ignored storage. Deleting a registry-managed model removes its owned files; reference models are never deleted.
