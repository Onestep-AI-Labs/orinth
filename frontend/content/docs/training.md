# Models & Training

Onestep AI Platform provides a robust local training ecosystem that executes long-running training jobs outside the FastAPI request path. Training jobs run in subprocesses, guaranteeing stability and continuous observability.

![Available Trained Models](/brand/5_available_trained_model_list.png)

## Available Models Zoo

The Models Zoo gives you access to a rich set of runnable model options:
- **Computer Vision:** Keras Applications (MobileNetV2, EfficientNet B0-B7, ResNet50, etc.) and Ultralytics YOLO families (YOLO11, YOLO26).
- **NLP Models:** Hugging Face transformers (BERT, BART) and Keras text classifiers.

## Training Workflows

- **Task-First Selection:** Training begins by selecting the task (e.g., Object Detection or Summarization). Model options and hyperparameters automatically filter to display only compatible choices for your dataset.
- **Observable Jobs:** A running training job streams logs in real-time, parsing metrics to provide epoch-by-epoch progress bars and estimated completion times.
- **Model Promotion:** Successfully completed runnable YOLO or Keras models are automatically registered as stable artifacts. You can then instantly select these newly promoted models for Testing and Inference.

![Training Details](/brand/6_training_details.png)

## Advanced Settings & Device Detection

- **Advanced parameters:** Beyond epochs, batch size, and learning rate, an advanced accordion exposes each model family's catalog defaults. Every value is submitted with the run, so a job is reproducible from its record alone.
- **Guardrails:** The learning-rate field warns when a value drifts outside the sane band for the selected base (fine-tuning transformers, for example, lives in a much narrower band than training from scratch).
- **Auto-detected accelerator:** Vision and LLM runs read the available device (CUDA, Apple MPS, or CPU) and default to it, so the shown configuration matches what the run will actually do.

## LLM Fine-tuning

LLM fine-tuning runs alongside the vision and NLP families. Choose an `llm_finetune` task, a Hugging Face base model (from the Hub or a local upload), and a method — LoRA, QLoRA, full fine-tune, or continued pretraining. See **LLM Fine-tuning & Chat** for the full walkthrough, including serving and GGUF export.

## Artifact Management

All logs, generated weights (e.g. `best.pt`, `best_model.keras`), configuration files, and `results.csv` files are stored securely under local ignored directories. Manual promotion remains idempotent for edge cases.
