"use client";

import { useState } from "react";
import Link from "next/link";
import { ButtonLink } from "@/features/platform/ui";
import { MarketingNav } from "./marketing-nav";
import { CodeBlock } from "./code-block";
import { SIGN_IN_HREF, MODELS_HREF } from "./routes";

const DOC_SECTIONS = [
  { id: "getting-started", label: "Getting Started", icon: "🚀" },
  { id: "architecture", label: "System Architecture", icon: "⚙️" },
  { id: "models-zoo", label: "Vision & NLP Model Zoo", icon: "🧠" },
  { id: "dataset-studio", label: "Dataset Studio & Recipes", icon: "🏷️" },
  { id: "api-reference", label: "REST API Reference", icon: "🔌" },
  { id: "local-deployment", label: "Local Deployment", icon: "💻" }
];

export function DocumentationPage() {
  const [activeSection, setActiveSection] = useState("getting-started");

  return (
    <div className="landing">
      <MarketingNav />

      <main>
        <div className="landing-container">
          <header className="docs-hero-header">
            <span className="landing-eyebrow">Documentation</span>
            <h1 className="landing-h1">Onestep AI Platform Documentation</h1>
            <p className="landing-lede">
              Technical specifications, model zoo architecture, Dataset Studio workflows, and REST API guides for local machine intelligence.
            </p>
          </header>

          <div className="docs-layout">
            {/* Left Sidebar Navigation */}
            <aside className="docs-sidebar">
              <nav className="docs-sidebar-nav" aria-label="Documentation Navigation">
                <span className="docs-sidebar-title">On This Page</span>
                <ul className="docs-sidebar-list">
                  {DOC_SECTIONS.map((sec) => (
                    <li key={sec.id}>
                      <a
                        href={`#${sec.id}`}
                        className={`docs-sidebar-link ${activeSection === sec.id ? "active" : ""}`}
                        onClick={() => setActiveSection(sec.id)}
                      >
                        <span className="docs-sidebar-icon">{sec.icon}</span>
                        {sec.label}
                      </a>
                    </li>
                  ))}
                </ul>

                <div className="docs-sidebar-divider" />

                <div className="docs-sidebar-cta">
                  <span className="docs-sidebar-cta-title">Try it in the Workspace</span>
                  <p className="docs-sidebar-cta-text">Launch your local project, annotate data, and run inference.</p>
                  <ButtonLink href={SIGN_IN_HREF} size="sm">
                    Open Platform →
                  </ButtonLink>
                </div>
              </nav>
            </aside>

            {/* Right Main Content Column */}
            <div className="docs-content">
              {/* Section 1: Getting Started */}
              <section id="getting-started" className="docs-section">
                <h2 className="docs-section-title">🚀 Getting Started</h2>
                <p className="docs-paragraph">
                  Onestep AI Platform is designed for 100% local operation. All model runs, training subprocesses, dataset annotations, and inference predictions run entirely on your own hardware without uploading data to external cloud APIs.
                </p>

                <div className="docs-info-box">
                  <span className="docs-info-title">💡 Core Architecture Principle</span>
                  <p className="docs-info-text">
                    Continuity is the organizing principle: an empty dataset points at upload, a finished training run points at testing, and a scored model points at inference.
                  </p>
                </div>

                <div className="docs-subblock">
                  <h3 className="docs-subblock-title">Quickstart Commands</h3>
                  <CodeBlock
                    code={`cp .env.example .env && cp frontend/.env.example frontend/.env.local\nmake doctor\nmake dev`}
                    label="quickstart commands"
                  />
                </div>
              </section>

              {/* Section 2: System Architecture */}
              <section id="architecture" className="docs-section">
                <h2 className="docs-section-title">⚙️ System Architecture</h2>
                <p className="docs-paragraph">
                  The system combines a decoupled Python 3.11 backend with a high-performance Next.js 15 frontend client.
                </p>

                <div className="docs-cards-grid">
                  <div className="docs-mini-card">
                    <span className="docs-mini-tag">Backend</span>
                    <h3 className="docs-mini-title">FastAPI &amp; SQLite</h3>
                    <p className="docs-mini-desc">
                      Asynchronous REST API endpoints, SQLAlchemy ORM persistence, lazy-loaded TensorFlow &amp; PyTorch predictors, and subprocess job runners.
                    </p>
                  </div>
                  <div className="docs-mini-card">
                    <span className="docs-mini-tag">Frontend</span>
                    <h3 className="docs-mini-title">Next.js 15 &amp; React 19</h3>
                    <p className="docs-mini-desc">
                      App router architecture, TypeScript strict checking, vanilla CSS design system tokens, and interactive canvas overlays.
                    </p>
                  </div>
                </div>
              </section>

              {/* Section 3: Vision & NLP Model Zoo */}
              <section id="models-zoo" className="docs-section">
                <h2 className="docs-section-title">🧠 Vision &amp; NLP Model Zoo</h2>
                <p className="docs-paragraph">
                  The catalog standardizes model inference across vision and language tasks into a single normalized output contract.
                </p>

                <div className="docs-cards-grid">
                  <div className="docs-mini-card">
                    <span className="docs-mini-tag">Computer Vision</span>
                    <h3 className="docs-mini-title">YOLOv11 &amp; U-Net / Inception</h3>
                    <p className="docs-mini-desc">
                      Pretrained Ultralytics YOLOv11 for instance segmentation and bounding boxes. Keras U-Net for 256x256 segmentation masks and Inception for 299x299 classification.
                    </p>
                    <Link href={MODELS_HREF} className="docs-link">View in Models Catalog →</Link>
                  </div>
                  <div className="docs-mini-card">
                    <span className="docs-mini-tag">NLP &amp; LLMs</span>
                    <h3 className="docs-mini-title">Transformers &amp; GGUF Serving</h3>
                    <p className="docs-mini-desc">
                      Hugging Face Transformers for text classification, summarization, and QA. Export LoRA/QLoRA fine-tuned weights to GGUF format for local CPU/GPU streaming chat.
                    </p>
                    <Link href={MODELS_HREF} className="docs-link">View in Models Catalog →</Link>
                  </div>
                </div>
              </section>

              {/* Section 4: Dataset Studio & Recipes */}
              <section id="dataset-studio" className="docs-section">
                <h2 className="docs-section-title">🏷️ Dataset Studio &amp; Recipes</h2>
                <p className="docs-paragraph">
                  Dataset Studio handles image polygon masks, text classification labels, and dataset split versions without overwriting raw source files.
                </p>
                <ul className="docs-bullet-list">
                  <li><strong>Hugging Face Hub Import:</strong> Search and import public datasets directly into your workspace.</li>
                  <li><strong>Data Recipes:</strong> Convert raw PDF/Markdown documents into structured instruction-tuning datasets.</li>
                  <li><strong>Reproducible Versioning:</strong> Freeze materialized dataset splits before kicking off training jobs.</li>
                </ul>
              </section>

              {/* Section 5: REST API Reference */}
              <section id="api-reference" className="docs-section">
                <h2 className="docs-section-title">🔌 REST API Reference</h2>
                <p className="docs-paragraph">
                  All platform capabilities are exposed via clean RESTful HTTP endpoints running on <code>http://localhost:8000/api/v1</code>.
                </p>

                <div className="landing-table-wrap">
                  <table className="landing-table">
                    <thead>
                      <tr>
                        <th scope="col">Method</th>
                        <th scope="col">Endpoint</th>
                        <th scope="col">Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><span className="docs-method-badge get">GET</span></td>
                        <th scope="row"><code>/api/v1/projects</code></th>
                        <td>List all active Vision &amp; NLP workspace projects.</td>
                      </tr>
                      <tr>
                        <td><span className="docs-method-badge post">POST</span></td>
                        <th scope="row"><code>/api/v1/projects</code></th>
                        <td>Create a new project workspace.</td>
                      </tr>
                      <tr>
                        <td><span className="docs-method-badge post">POST</span></td>
                        <th scope="row"><code>/api/v1/datasets/upload</code></th>
                        <td>Upload raw image or document dataset files.</td>
                      </tr>
                      <tr>
                        <td><span className="docs-method-badge post">POST</span></td>
                        <th scope="row"><code>/api/v1/training/runs</code></th>
                        <td>Start an asynchronous PyTorch/Keras subprocess training job.</td>
                      </tr>
                      <tr>
                        <td><span className="docs-method-badge post">POST</span></td>
                        <th scope="row"><code>/api/v1/inference/predict</code></th>
                        <td>Execute single-item inference with polygon segmentation overlay.</td>
                      </tr>
                      <tr>
                        <td><span className="docs-method-badge post">POST</span></td>
                        <th scope="row"><code>/api/v1/llm/chat</code></th>
                        <td>Stream tokens from a locally served GGUF language model.</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </section>

              {/* Section 6: Local Deployment */}
              <section id="local-deployment" className="docs-section">
                <h2 className="docs-section-title">💻 Local Deployment &amp; Hardware Requirements</h2>
                <p className="docs-paragraph">
                  The platform runs cross-platform on macOS (Apple Silicon / Intel), Linux, and Windows.
                </p>
                <div className="docs-info-box">
                  <span className="docs-info-title">🖥️ Recommended Hardware Specs</span>
                  <p className="docs-info-text">
                    Python 3.11, 16GB+ RAM (Unified Memory or System RAM), 10GB free disk space for model weights. GPU acceleration supported via PyTorch Metal (macOS MPS), CUDA (NVIDIA), or ROCm.
                  </p>
                </div>
              </section>
            </div>
          </div>
        </div>
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
