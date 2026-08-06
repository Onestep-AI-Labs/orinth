export interface SystemModel {
  id: string;
  name: string;
  family: string;
  taskType: "segmentation" | "classification" | "summarization" | "question_answering" | "llm" | "architecture";
  taskLabel: string;
  description: string;
  longDescription: string;
  format: string;
  size: string;
  metrics: Record<string, string>;
  inputFormat: string;
  outputFormat: string;
  tags: string[];
  features: string[];
  codeExample: string;
}

export const SYSTEM_MODELS: SystemModel[] = [
  {
    id: "yolo_11_best",
    name: "YOLOv11 Best",
    family: "YOLO (Ultralytics)",
    taskType: "segmentation",
    taskLabel: "Instance Segmentation",
    description: "Real-time single-stage object and region instance segmentation model.",
    longDescription:
      "Trained on high-resolution image datasets, YOLOv11 Best produces precise polygon segmentation masks and confidence scores in a single forward pass. Optimized for local execution with Ultralytics PyTorch backend.",
    format: "PyTorch (.pt)",
    size: "42.8 MB",
    metrics: {
      "mAP@50": "0.892",
      "mAP@50-95": "0.714",
      Precision: "0.915",
      Recall: "0.868",
      "Inference Speed": "12 ms / frame"
    },
    inputFormat: "RGB Image (JPG, PNG, WebP) · Any resolution",
    outputFormat: "Polygon coordinate masks, bounding boxes, class labels & confidence score (0.0 - 1.0)",
    tags: ["Computer Vision", "Segmentation", "Real-Time", "PyTorch", "Ultralytics"],
    features: [
      "Sub-15ms real-time inference latency",
      "Exact polygon boundaries for complex overlapping regions",
      "Default confidence threshold 0.65, IoU threshold 0.70",
      "Exportable to ONNX, TensorRT, and CoreML"
    ],
    codeExample: `from app.ml.vision.yolo.predictor import YoloPredictor

predictor = YoloPredictor(weights_path="models/yolo_11_best/weights/best.pt")
results = predictor.predict(image_bytes=raw_image_data, conf_threshold=0.65)
print(f"Detected {len(results.masks)} segmentation regions.")`
  },
  {
    id: "unet_inception",
    name: "U-Net + Inception Hybrid",
    family: "Keras / TensorFlow",
    taskType: "segmentation",
    taskLabel: "Segmentation & Classification",
    description: "Two-stage deep learning pipeline combining U-Net lesion segmentation with Inception classification.",
    longDescription:
      "A dual-architecture pipeline designed for detailed image understanding. The U-Net stage performs dense 256x256 pixel-level segmentation, passing isolated regions into an Inception v3 classifier (299x299) for multi-class evaluation.",
    format: "Keras (.keras)",
    size: "128.4 MB",
    metrics: {
      "Dice Score": "0.874",
      IoU: "0.776",
      "Classification Accuracy": "94.2%",
      "Macro F1": "0.938",
      "Input Resolution": "256x256 / 299x299"
    },
    inputFormat: "Normalized RGB Image Tensor (256x256x3)",
    outputFormat: "Binary mask matrix (256x256) + 5-class probability vector",
    tags: ["Computer Vision", "U-Net", "Inception", "Keras", "TensorFlow"],
    features: [
      "Two-stage pipeline: spatial mask localization + high-capacity classification",
      "Skip-connections preserve fine geometric details across encoder/decoder",
      "Compatible with TensorFlow/Keras 3 multi-backend engine",
      "Integrated metrics tracking for Dice coefficient and Jaccard index"
    ],
    codeExample: `from app.ml.vision.unet_inception.predictor import UnetInceptionPredictor

predictor = UnetInceptionPredictor(
    unet_path="models/unet_inception/best_unet_model.keras",
    classifier_path="models/unet_inception/best_classifier_inception.keras"
)
output = predictor.predict(image_bytes=raw_image_data)`
  },
  {
    id: "keyword_text_classifier",
    name: "Keyword Text Classifier",
    family: "NLP Baseline",
    taskType: "classification",
    taskLabel: "Text Classification",
    description: "High-speed TF-IDF and keyword matching baseline classifier for text documents.",
    longDescription:
      "Lightweight CPU-friendly text classification engine. Utilizes TF-IDF n-gram vectorization and linear decision boundaries to provide instantaneous sentiment and category scoring without deep learning overhead.",
    format: "Scikit-Learn / Python",
    size: "1.2 MB",
    metrics: {
      Accuracy: "86.5%",
      "Macro F1": "0.842",
      Precision: "0.858",
      Recall: "0.835",
      Latency: "<1 ms"
    },
    inputFormat: "Plain text string or structured document payload",
    outputFormat: "Predicted class label + probability distribution per class",
    tags: ["NLP", "Classification", "TF-IDF", "Baseline", "Lightweight"],
    features: [
      "Zero GPU memory required — ultra-low latency execution",
      "Deterministic and fully interpretable feature importance scores",
      "Supports custom vocabulary dictionary overrides",
      "Ideal baseline benchmark for complex NLP tasks"
    ],
    codeExample: `from app.ml.nlp.baseline.predictors import TextClassificationPredictor

predictor = TextClassificationPredictor(labels=["positive", "negative", "neutral"])
result = predictor.predict("The model training converged cleanly with zero errors.")
print(result.label, result.confidence)`
  },
  {
    id: "extractive_summarizer",
    name: "Extractive Text Summarizer",
    family: "NLP Baseline",
    taskType: "summarization",
    taskLabel: "Text Summarization",
    description: "Graph-based TextRank and centrality summarizer for documents and reports.",
    longDescription:
      "Identifies key sentences from technical documents, articles, and logs using sentence graph centrality and cosine vector similarity. Delivers precise, non-hallucinated summaries directly sourced from original text.",
    format: "Python Native",
    size: "0.5 MB",
    metrics: {
      "ROUGE-1": "42.1",
      "ROUGE-2": "19.4",
      "ROUGE-L": "38.6",
      Latency: "3 ms",
      "Compression Ratio": "75%"
    },
    inputFormat: "Multi-paragraph text document or long article",
    outputFormat: "Ranked list of key sentences forming a coherent summary",
    tags: ["NLP", "Summarization", "TextRank", "Extractive", "No Hallucination"],
    features: [
      "100% faithful to source document — zero risk of language model hallucination",
      "Configurable sentence count and token budget target",
      "Preserves original technical terminology and numerical figures",
      "Instantaneous CPU execution for bulk document processing"
    ],
    codeExample: `from app.ml.nlp.baseline.predictors import ExtractiveSummarizerPredictor

summarizer = ExtractiveSummarizerPredictor(max_sentences=3)
summary = summarizer.predict(text_content=document_text)`
  },
  {
    id: "keyword_qa",
    name: "Keyword Overlap Question Answering",
    family: "NLP Baseline",
    taskType: "question_answering",
    taskLabel: "Question Answering",
    description: "Information retrieval QA engine using passage scoring and n-gram keyword overlap.",
    longDescription:
      "Answering questions over structured documentation by scoring candidate text spans against the user query. Combines BM25 token overlap with lexical matching to return verbatim answers and context passages.",
    format: "Python Native",
    size: "0.8 MB",
    metrics: {
      "Exact Match": "78.4%",
      "F1-Score": "0.831",
      Latency: "2 ms",
      "Passage Recall": "91.2%"
    },
    inputFormat: "Context text / document + User question string",
    outputFormat: "Extracted answer span string, start/end character offsets, and confidence score",
    tags: ["NLP", "Question Answering", "Retrieval", "BM25", "Search"],
    features: [
      "Verbatim span extraction with exact document offsets",
      "No GPU required — fast search over internal knowledge bases",
      "Supports multi-passage candidate ranking",
      "Provides grounding citations back to source paragraphs"
    ],
    codeExample: `from app.ml.nlp.baseline.predictors import KeywordQAPredictor

qa = KeywordQAPredictor()
answer = qa.predict(context=document, question="What optimizer was used?")`
  },
  {
    id: "hf_transformers_nlp",
    name: "Hugging Face Transformers Suite",
    family: "Transformers (PyTorch)",
    taskType: "classification",
    taskLabel: "NLP Transformers",
    description: "Local fine-tuning & inference pipeline for BERT, RoBERTa, BART, and DeBERTa.",
    longDescription:
      "Comprehensive Hugging Face Transformers pipeline for sequence classification, token tagging, and generative summarization. Supports local model weights and private Hugging Face Hub repositories with custom tokenizers.",
    format: "Hugging Face PyTorch (safetensors)",
    size: "440 MB",
    metrics: {
      "BERT Accuracy": "92.8%",
      "RoBERTa F1": "0.941",
      "BART ROUGE-L": "44.5",
      "QA F1": "88.9%"
    },
    inputFormat: "Raw text string / tokenized input IDs and attention masks",
    outputFormat: "Classification logits, generated text sequences, or span logits",
    tags: ["NLP", "Transformers", "BERT", "RoBERTa", "Hugging Face", "PyTorch"],
    features: [
      "Seamless integration with Hugging Face Hub dataset import",
      "Automated evaluation with accuracy, F1, and ROUGE metrics",
      "Support for quantized INT8 / FP16 model weights",
      "Local fine-tuning with gradient accumulation and learning rate decay"
    ],
    codeExample: `from app.ml.nlp.huggingface.predictors import HuggingFaceTextClassificationPredictor

predictor = HuggingFaceTextClassificationPredictor(model_name_or_path="bert-base-uncased")
prediction = predictor.predict("Model evaluation completed with 94% accuracy.")`
  },
  {
    id: "llm_gguf_chat",
    name: "Served LLM GGUF & LoRA",
    family: "LLM (Llama.cpp / GGUF)",
    taskType: "llm",
    taskLabel: "Language Models & Chat",
    description: "Quantized GGUF language models served locally with sampler controls, search, and reasoning.",
    longDescription:
      "Instruction-tuned language models (Llama 3, Gemma 2/4, Qwen 2.5) fine-tuned via LoRA/QLoRA and exported to GGUF format. Served locally with an embedded llama.cpp backend featuring live web search citations, customizable system prompts, temperature/top_p sampling, and visual reasoning inspection.",
    format: "GGUF (.gguf)",
    size: "4.2 GB (Q4_K_M)",
    metrics: {
      "Generation Speed": "34.2 tokens / sec",
      "Context Window": "8,192 tokens",
      "VRAM Usage": "4.2 GB",
      "Quantization": "Q4_K_M"
    },
    inputFormat: "Prompt conversation history, system prompt, temperature, top_k/top_p",
    outputFormat: "Streamed text response with reasoning steps & web search source citations",
    tags: ["LLM", "GGUF", "LoRA", "Llama.cpp", "Chat", "Web Search"],
    features: [
      "Local high-performance GGUF serving powered by llama.cpp",
      "LoRA / QLoRA / Full fine-tuning runners with automatic accelerator detection",
      "Integrated web search mode with inline domain citations",
      "Real-time token streaming with step-by-step reasoning panel"
    ],
    codeExample: `# Async HTTP stream to served GGUF model endpoint
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/serving/chat",
        json={
            "model": "llama-3-8b-instruct.gguf",
            "messages": [{"role": "user", "content": "Explain LoRA fine-tuning."}],
            "temperature": 0.7
        }
    )`
  },
  {
    id: "architecture_studio_custom",
    name: "Architecture Studio Visual Models",
    family: "Custom Architecture Studio",
    taskType: "architecture",
    taskLabel: "Custom Neural Graph",
    description: "Visual block-based model graph editor emitting clean Keras 3 and PyTorch code.",
    longDescription:
      "Design custom deep learning model architectures visually using drag-and-drop layer blocks (ResNet blocks, Attention heads, RMSNorm, Conv2D, Dense, Transformer Encoder/Decoder). Generates executable Python code for Keras 3 and PyTorch with automatic shape validation.",
    format: "Keras 3 / PyTorch (.py)",
    size: "Code Generator",
    metrics: {
      "Supported Layers": "25+ Building Blocks",
      "Code Emitter": "Keras 3 + PyTorch 2",
      "Presets": "ViT, ResNet, UNet, Gemma 4, Qwen",
      "Validation": "Real-time Shape Checking"
    },
    inputFormat: "Visual node graph schema JSON",
    outputFormat: "Executable Python model module code (nn.Module or tf.keras.Model)",
    tags: ["Architecture Studio", "Keras", "PyTorch", "Visual Editor", "Code Emitter"],
    features: [
      "n8n-style interactive node canvas with drag-and-drop layer palette",
      "One-click python code export for PyTorch 2 and Keras 3",
      "Pre-configured architecture presets: Vision Transformers, ResNet, UNet, LLM blocks",
      "Automatic dimension check and tensor shape mismatch warnings"
    ],
    codeExample: `# Emitted PyTorch code from Architecture Studio
import torch
import torch.nn as nn

class CustomVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm1(self.conv1(x)))`
  }
];
