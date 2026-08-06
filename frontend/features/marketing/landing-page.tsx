import Image from "next/image";
import Link from "next/link";
import { ButtonLink } from "@/features/platform/ui";
import { CodeBlock } from "./code-block";
import { MarketingNav } from "./marketing-nav";
import { PopularModelsSlider } from "./popular-models-slider";
import { SIGN_IN_HREF } from "./routes";

const REPO_HREF = "https://github.com/L007/onestep-ai-platform";
const DOWNLOAD_HREF = "https://github.com/L007/onestep-ai-platform/archive/refs/heads/main.zip";

const CMD_STEP_1 = `cp .env.example .env && cp frontend/.env.example frontend/.env.local`;
const CMD_STEP_2 = `make doctor`;
const CMD_STEP_3 = `make dev`;

/**
 * Every claim on this page is traceable to README.md or docs/ai/rules.md. No
 * invented metrics, no testimonials, no logo wall — there are no users to quote
 * yet, and fabricating them is the fastest way to make an honest project read
 * as dishonest. Captures are the real app, unretouched.
 */
const STAGES = [
  {
    n: "01",
    title: "Organize & Scope",
    subtitle: "Say goodbye to notebook clutter and loose script files.",
    action: "Create isolated projects for Vision & NLP work to manage datasets, runs, and checkpoints.",
    result: "🗂️ Clean, organized project workspace with strict version boundary.",
    tags: ["Projects", "Vision & NLP Workspace"],
    image: "/brand/1_project_list.png",
    caption: "Project list & scope manager"
  },
  {
    n: "02",
    title: "Prepare & Annotate",
    subtitle: "Label data like a pro — zero cloud upload required.",
    action: "Annotate polygon segmentation masks, tag classes, or import Hugging Face Hub datasets & recipes.",
    result: "🏷️ Gold-standard ground truth dataset version ready for training.",
    tags: ["Dataset Studio", "Polygon Masks", "Hub Import", "Data Recipes"],
    image: "/brand/5_dataset_studio_nlp_clasification_task.png",
    caption: "Dataset Studio — annotation & split manager"
  },
  {
    n: "03",
    title: "Train & Fine-Tune",
    subtitle: "Press start, grab a coffee, watch epoch curves live.",
    action: "Launch single-click PyTorch or Keras runs (YOLO, U-Net, BERT, LoRA LLMs) in background subprocesses.",
    result: "⚡ High-accuracy model weights & real-time loss tracking.",
    tags: ["Subprocess Runs", "Ultralytics & Keras", "LoRA Fine-Tuning"],
    image: "/brand/6_training_details.png",
    caption: "Training detail & live metric loss curve"
  },
  {
    n: "04",
    title: "Test & Benchmark",
    subtitle: "No guessing — read real numbers behind model behavior.",
    action: "Score trained checkpoints against held-out validation splits with task-aware metrics & ROC curves.",
    result: "📊 Verified model scorecard & per-item error breakdown.",
    tags: ["Task-Aware Metrics", "mAP / F1 / Dice", "Per-Item Analysis"],
    image: "/brand/5_available_trained_model_list.png",
    caption: "Trained models catalog & evaluation scorecard"
  },
  {
    n: "05",
    title: "Inspect & Serve",
    subtitle: "Run instant predictions or chat with your local LLM.",
    action: "Execute single-item inference with polygon overlays or stream chat with local GGUF models.",
    result: "🚀 Live visual predictions, chat responses, and local REST API.",
    tags: ["Visual Overlays", "GGUF Chat", "Local REST API"],
    image: "/brand/7_inference_image_segmentation_task.png",
    caption: "Inference studio — image segmentation overlay"
  }
];

export function LandingPage() {
  return (
    <div className="landing">
      <MarketingNav />

      <main>
        <div className="landing-container">
          <section className="landing-hero landing-hero-centered">
            <div className="landing-hero-content">
              <span className="landing-eyebrow">Open source · Local & Self-Hosted</span>

              <h1 className="landing-h1">
                Run, train, and evaluate AI models on your own machine.
              </h1>

              <p className="landing-lede">
                Onestep AI Platform is an open-source, easy-to-use local workspace for Computer Vision,
                NLP, and LLM intelligence.
              </p>

              <div className="landing-cta-row landing-cta-row-centered">
                <ButtonLink variant="secondary" href={REPO_HREF} target="_blank" rel="noreferrer">
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    style={{ marginRight: "8px", display: "inline-block", verticalAlign: "middle" }}
                  >
                    <path
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                    />
                  </svg>
                  GitHub Source
                </ButtonLink>

                <ButtonLink href={DOWNLOAD_HREF} target="_blank" rel="noreferrer">
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    style={{ marginRight: "8px", display: "inline-block", verticalAlign: "middle" }}
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  Download
                </ButtonLink>
              </div>

              <div className="landing-trust-pills">
                <span className="landing-trust-pill">🔒 100% Local & Private</span>
                <span className="landing-trust-pill">⚡ Zero Cloud Dependency</span>
                <span className="landing-trust-pill">📜 Open Source (Apache 2.0)</span>
                <span className="landing-trust-pill">⚒️ Easy to configure and run</span>
              </div>
            </div>
          </section>
        </div>

        <section id="models" className="landing-section">
          <div className="landing-container landing-container-centered">
            <h2 className="landing-featured-title">Featured Models</h2>
            <PopularModelsSlider />
          </div>
        </section>

        <section id="workflow" className="landing-section workflow-section">
          <div className="landing-container">
            <div className="workflow-header">
              <span className="landing-eyebrow">The Workflow</span>
              <h2 className="landing-h2">From Raw Data to Deployed Inference in 5 Simple Steps</h2>
              <p className="landing-lede">
                Continuity is the organizing principle: every stage seamlessly hands off to the next — from raw dataset upload to measured model predictions.
              </p>
            </div>

            <div className="workflow-timeline-wrapper">
              <ol className="workflow-timeline">
                {STAGES.map((stage, idx) => (
                  <li className="workflow-timeline-item" key={stage.n}>
                    <div className="workflow-timeline-rail">
                      <div className="workflow-timeline-node">
                        <span className="workflow-timeline-number">{stage.n}</span>
                      </div>
                      {idx < STAGES.length - 1 && <div className="workflow-timeline-line" />}
                    </div>

                    <div className="workflow-timeline-content">
                      <div className="workflow-stage-card">
                        <div className="workflow-stage-body">
                          <span className="workflow-stage-step-pill">Stage {stage.n}</span>
                          <h3 className="workflow-stage-title">{stage.title}</h3>
                          <p className="workflow-stage-subtitle">{stage.subtitle}</p>

                          <div className="workflow-action-result-box">
                            <div className="workflow-flow-step">
                              <span className="workflow-flow-lbl">1. What You Do</span>
                              <p className="workflow-flow-txt">{stage.action}</p>
                            </div>
                            <div className="workflow-flow-arrow">↓</div>
                            <div className="workflow-flow-step workflow-flow-result">
                              <span className="workflow-flow-lbl">2. What You Get</span>
                              <p className="workflow-flow-txt">{stage.result}</p>
                            </div>
                          </div>

                          <ul className="workflow-stage-tags">
                            {stage.tags.map((tag) => (
                              <li key={tag}>{tag}</li>
                            ))}
                          </ul>
                        </div>

                        {/* Image padded & framed cleanly inside container card */}
                        <div className="workflow-image-card-container">
                          <figure className="workflow-image-card">
                            <div className="workflow-image-frame">
                              <Image
                                src={stage.image}
                                alt={`${stage.title} stage — ${stage.caption}`}
                                width={1915}
                                height={927}
                                loading="lazy"
                                sizes="(max-width: 60rem) 100vw, 42vw"
                              />
                            </div>
                            <figcaption>{stage.caption}</figcaption>
                          </figure>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </section>

        <section id="quickstart" className="landing-section quickstart-section">
          <div className="landing-container">
            <div className="quickstart-header">
              <span className="landing-eyebrow">Quickstart Setup</span>
              <h2 className="landing-h2">Run the Full Stack in 3 Easy Steps</h2>
              <p className="landing-lede">
                A FastAPI & SQLite backend, a Next.js frontend, and single-click <code>make</code> targets. Copy each command independently to get up and running on your local machine.
              </p>
            </div>

            <div className="quickstart-grid">
              <div className="quickstart-card">
                <div className="quickstart-card-header">
                  <span className="quickstart-step-num">Step 01</span>
                  <h3 className="quickstart-card-title">Configure Environment Variables</h3>
                </div>
                <p className="quickstart-card-desc">
                  Copy template environment settings for FastAPI backend (<code>:8000</code>) and Next.js frontend (<code>:3000</code>).
                </p>
                <CodeBlock code={CMD_STEP_1} label="step 1 environment setup command" />
              </div>

              <div className="quickstart-card">
                <div className="quickstart-card-header">
                  <span className="quickstart-step-num">Step 02</span>
                  <h3 className="quickstart-card-title">Verify Prerequisites</h3>
                </div>
                <p className="quickstart-card-desc">
                  Check local Python 3.11, Node.js, <code>uv</code>, and <code>pnpm</code> dependencies automatically.
                </p>
                <CodeBlock code={CMD_STEP_2} label="step 2 doctor command" />
              </div>

              <div className="quickstart-card">
                <div className="quickstart-card-header">
                  <span className="quickstart-step-num">Step 03</span>
                  <h3 className="quickstart-card-title">Launch Platform Stack</h3>
                </div>
                <p className="quickstart-card-desc">
                  Start backend (port 8000) and frontend (port 3000) concurrently with interleaved terminal logs.
                </p>
                <CodeBlock code={CMD_STEP_3} label="step 3 launch stack command" />
              </div>
            </div>
          </div>
        </section>

        <section className="landing-close">
          <div className="landing-container landing-close-container">
            <span className="landing-eyebrow">Get Started</span>
            <h2 className="landing-h1 landing-close-title">Start with a project.</h2>
            <p className="landing-lede landing-close-lede">
              Create a workspace project, point it at a dataset, and let the platform guide you from annotation to measured model inference.
            </p>
            <div className="landing-cta-row landing-cta-row-centered">
              <ButtonLink href={SIGN_IN_HREF} className="landing-close-btn">
                Open the Platform →
              </ButtonLink>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div className="landing-container landing-footer-simple">
          <p className="landing-footer-text">
            Onestep AI Platform — Research and engineering workspace for Vision &amp; NLP intelligence.
          </p>
        </div>
      </footer>
    </div>
  );
}
