import type { Step } from "react-joyride";

/**
 * Guided-tour definitions for every main platform surface, keyed by a tour id.
 *
 * Steps anchor to `[data-tour="…"]` hooks placed on stable structural elements
 * in the shell and pages — never on text or class names that churn. A tour only
 * runs on the route where its anchors are mounted (see `tourIdForPath`), so a
 * missing target means an anchor drifted, not a normal state.
 *
 * Copy follows DESIGN.md §1 voice: operational, noun labels, verb actions, no
 * marketing tone. Every claim traces to a real control on the screen.
 */
export type TourDefinition = {
  /** Human label used by the launcher button and telemetry. */
  label: string;
  steps: Step[];
};

const projects: TourDefinition = {
  label: "Workspace tour",
  steps: [
    {
      target: '[data-tour="brand"]',
      title: "Welcome to Orinth",
      content:
        "A local studio for vision and NLP. The workflow is always: Organize → Prepare → Train → Test → Inspect. This quick tour shows where each stage lives.",
      placement: "right"    },
    {
      target: '[data-tour="projects-new"]',
      title: "Create a project",
      content:
        "A project scopes its own datasets, runs, and the models promoted out of them. Vision and NLP work stay separate here.",
      placement: "bottom"
    },
    {
      target: '[data-tour="projects-grid"]',
      title: "Open a workspace",
      content:
        "Pick a project to drop into its workspace. That reveals the project rail — Datasets, Models, Training, Testing, and Inference.",
      placement: "top"
    },
    {
      target: '[data-tour="nav-settings"]',
      title: "Keys and settings",
      content:
        "Add a Hugging Face token for gated base models and an OpenRouter key for LLM-assisted record generation.",
      placement: "right"
    }
  ]
};

const datasets: TourDefinition = {
  label: "Datasets tour",
  steps: [
    {
      target: '[data-tour="datasets-catalog"]',
      title: "Dataset catalog",
      content:
        "Every dataset in this project, grouped by origin: your own, HuggingFace imports, and read-only shared samples.",
      placement: "bottom"    },
    {
      target: '[data-tour="dataset-ingest"]',
      title: "Start with your data",
      content:
        "Drop a folder or a pile of files — images, CSV, JSONL, Parquet. Orinth works out the " +
        "task, the labels, and the splits. Nothing to declare first.",
      placement: "bottom"
    },
    {
      target: '[data-tour="datasets-import"]',
      title: "Or somewhere else",
      content:
        "Same panel, different source: pull a dataset from the HuggingFace Hub, build one from " +
        "your own PDFs and documents, or start empty and upload into it yourself.",
      placement: "left"
    },
    {
      target: '[data-tour="datasets-card"]',
      title: "Open Dataset Studio",
      content:
        "Open a dataset to see what Orinth made of it, browse and label items, and adjust how it " +
        "is prepared.",
      placement: "top"
    }
  ]
};

const datasetStudio: TourDefinition = {
  label: "Dataset Studio tour",
  steps: [
    {
      target: '[data-tour="dataset-studio-header"]',
      title: "Dataset Studio",
      content: "You've opened a dataset. Prepare it here, then head back to the catalog when you're done.",
      placement: "bottom"
    },
    {
      target: '[data-tour="dataset-studio-overview"]',
      title: "Overview",
      content:
        "What Orinth made of your data, and whether it can be trained on yet. Every decision " +
        "shows what it was based on, and you can undo the lot.",
      placement: "bottom"
    },
    {
      target: '[data-tour="dataset-studio-items"]',
      title: "Data",
      content: "Browse, upload, and label the items — the annotation editor opens beside them.",
      placement: "bottom"
    },
    {
      target: '[data-tour="dataset-studio-config"]',
      title: "Prepare",
      content:
        "Class balance and split counts, next to the preprocessing and splits that produced " +
        "them. Orinth fills these in; change anything you disagree with.",
      placement: "bottom"
    }
  ]
};

const training: TourDefinition = {
  label: "Training tour",
  steps: [
    {
      target: '[data-tour="training-model"]',
      title: "Pick a task and base model",
      content:
        "Choose a task, then a base model from the zoo. Its defaults fill in the parameters below.",
      placement: "bottom"    },
    {
      target: '[data-tour="training-dataset"]',
      title: "Choose a prepared dataset",
      content: "Only dataset versions that match the selected task appear here.",
      placement: "right"
    },
    {
      target: '[data-tour="training-parameters"]',
      title: "Tune parameters",
      content:
        "Epochs, batch size, learning rate, and task-specific knobs. The advanced accordion holds the rest.",
      placement: "left"
    },
    {
      target: '[data-tour="training-run"]',
      title: "Prepare and start",
      content:
        "Prepare downloads a base model if needed; Start training launches the run as a subprocess, so a long fine-tune never blocks the API.",
      placement: "top"
    },
    {
      target: '[data-tour="training-jobs"]',
      title: "Track runs",
      content: "Live and finished runs land here. Open one for its progress, logs, and metrics.",
      placement: "top"
    }
  ]
};

const testing: TourDefinition = {
  label: "Testing tour",
  steps: [
    {
      target: '[data-tour="testing-new"]',
      title: "Score a model",
      content:
        "Pick a task, one or more trained models, and a held-out dataset split. Testing runs task-aware metrics — accuracy, ROUGE, exact match, and more.",
      placement: "right"    },
    {
      target: '[data-tour="testing-jobs"]',
      title: "Evaluation runs",
      content: "Each evaluation lands here. Open one to read the per-item results sitting behind every number.",
      placement: "left"
    },
    {
      target: '[data-tour="testing-comparison"]',
      title: "Compare side by side",
      content: "Test several models against one dataset to compare them directly in this panel.",
      placement: "top"
    }
  ]
};

const inference: TourDefinition = {
  label: "Inference tour",
  steps: [
    {
      target: '[data-tour="inference-run"]',
      title: "Run a single item",
      content:
        "Choose a task and model, drop in an image or text, and run. Detection tasks expose confidence and IoU controls.",
      placement: "right"    },
    {
      target: '[data-tour="inference-result"]',
      title: "Read the output",
      content: "Overlays, class confidence, and detections. The shape is the same across model families.",
      placement: "left"
    },
    {
      target: '[data-tour="inference-history"]',
      title: "Run history",
      content: "Every past run is here — select a row to bring its result back into the panel above.",
      placement: "top"
    }
  ]
};

const chat: TourDefinition = {
  label: "Chat tour",
  steps: [
    {
      target: '[data-tour="chat-serving"]',
      title: "Serve a model",
      content:
        "Start a GGUF model here to bring the runtime up. Switch or stop the served model at any time.",
      placement: "right"    },
    {
      target: '[data-tour="chat-sampler"]',
      title: "Sampler and system prompt",
      content: "Temperature, top-p, and max tokens shape each reply; the system prompt sets behavior for the conversation.",
      placement: "right"
    },
    {
      target: '[data-tour="chat-transcript"]',
      title: "Transcript",
      content:
        "Your conversation streams here, with token throughput below. A research instrument — not a medical assistant.",
      placement: "left"
    },
    {
      target: '[data-tour="chat-composer"]',
      title: "Send a message",
      content:
        "Type and press Enter to send. Toggle web search to ground answers with cited sources, or Thinking to show the model's reasoning.",
      placement: "top"
    }
  ]
};

const models: TourDefinition = {
  label: "Models tour",
  steps: [
    {
      target: '[data-tour="models-catalog"]',
      title: "Model catalog",
      content:
        "Reference, trained, and uploaded models, grouped by source. Open a card for detail, export, and serving.",
      placement: "bottom"    },
    {
      target: '[data-tour="models-upload"]',
      title: "Upload custom weights",
      content: "Bring your own model — a checkpoint, a GGUF file, or a LoRA adapter — into the catalog.",
      placement: "left"
    }
  ]
};

const settings: TourDefinition = {
  label: "Settings tour",
  steps: [
    {
      target: '[data-tour="settings-hf"]',
      title: "Hugging Face token",
      content:
        "Save a Hub token to download gated base models. It is stored server-side and never returned to the browser.",
      placement: "bottom"    },
    {
      target: '[data-tour="settings-openrouter"]',
      title: "OpenRouter key",
      content:
        "Powers LLM-assisted record generation in data recipes. Without a key, recipes fall back to deterministic rule-based records.",
      placement: "top"
    }
  ]
};

export const TOURS: Record<string, TourDefinition> = {
  projects,
  datasets,
  "dataset-studio": datasetStudio,
  training,
  testing,
  inference,
  chat,
  models,
  settings
};

export type TourId = keyof typeof TOURS;

/**
 * Resolve which tour, if any, belongs to a pathname. Only the surfaces whose
 * anchors are actually mounted map to a tour — detail views (`/training/[id]`),
 * the recipe builder (`/datasets/recipes`), and project create/settings do not,
 * so the launcher stays hidden rather than pointing at absent targets.
 */
export function tourIdForPath(pathname: string, hasDatasetParam = false): TourId | null {
  if (pathname === "/projects") return "projects";
  if (pathname === "/inference/chat") return "chat";
  if (pathname === "/inference") return "inference";
  // `/datasets` is both the catalog and, with `?dataset=…`, the open Dataset
  // Studio — two different views with different anchors, so two tours.
  if (pathname === "/datasets") return hasDatasetParam ? "dataset-studio" : "datasets";
  if (pathname === "/training") return "training";
  if (pathname === "/testing") return "testing";
  if (pathname === "/models") return "models";
  if (pathname.startsWith("/settings")) return "settings";
  return null;
}
