"use client";

import { useState } from "react";
import Link from "next/link";
import { ButtonLink } from "@/features/platform/ui";
import { MarketingNav } from "./marketing-nav";
import { SYSTEM_MODELS, type SystemModel } from "./models-data";
import { SIGN_IN_HREF } from "./routes";

export function ModelsPage() {
  const [activeFilter, setActiveFilter] = useState<string>("all");

  const filteredModels = SYSTEM_MODELS.filter((model) => {
    if (activeFilter === "all") return true;
    if (activeFilter === "vision") return model.taskType === "segmentation" || model.taskType === "classification";
    if (activeFilter === "nlp") return model.taskType === "classification" || model.taskType === "summarization" || model.taskType === "question_answering";
    if (activeFilter === "llm") return model.taskType === "llm";
    if (activeFilter === "architecture") return model.taskType === "architecture";
    return true;
  });

  return (
    <div className="landing">
      <MarketingNav />

      <main>
        <div className="landing-container">
          <section className="landing-hero" style={{ paddingBottom: "2rem" }}>
            <div>
              <span className="landing-eyebrow">Available Models Catalog</span>
              <h1 className="landing-h1">Models available across Vision, NLP & LLMs.</h1>
              <p className="landing-lede">
                Explore reference weights, pretrained baselines, fine-tuned transformers, GGUF chat models,
                and custom neural architectures available in Onestep AI Platform.
              </p>
              <div className="landing-cta-row">
                <ButtonLink href={SIGN_IN_HREF}>Open in Workspace</ButtonLink>
              </div>
            </div>
          </section>

          <div className="models-filter-bar">
            <span className="models-filter-label">Filter by domain:</span>
            <div className="models-filter-buttons">
              {[
                { id: "all", label: "All Models" },
                { id: "vision", label: "Computer Vision" },
                { id: "nlp", label: "NLP Baselines" },
                { id: "llm", label: "LLMs & Serving" },
                { id: "architecture", label: "Architecture Studio" }
              ].map((filter) => (
                <button
                  key={filter.id}
                  type="button"
                  className={`models-filter-btn ${activeFilter === filter.id ? "active" : ""}`}
                  onClick={() => setActiveFilter(filter.id)}
                >
                  {filter.label}
                </button>
              ))}
            </div>
          </div>

          <div className="models-grid">
            {filteredModels.map((model) => (
              <article key={model.id} className="model-card">
                <div className="model-card-header">
                  <span className="model-card-badge">{model.taskLabel}</span>
                  <span className="model-card-format">{model.format}</span>
                </div>

                <div className="model-card-main">
                  <h2 className="model-card-title">
                    <Link href={`/model-catalog/${model.id}`}>{model.name}</Link>
                  </h2>
                  <span className="model-card-family">Family: {model.family}</span>
                  <p className="model-card-desc">{model.description}</p>
                </div>

                <div className="model-card-bottom">
                  <div className="model-card-metrics">
                    {Object.entries(model.metrics).slice(0, 3).map(([key, val]) => (
                      <div key={key} className="model-card-metric-item">
                        <span className="model-card-metric-label">{key}</span>
                        <strong className="model-card-metric-value">{val}</strong>
                      </div>
                    ))}
                  </div>

                  <div className="model-card-tags">
                    {model.tags.map((tag) => (
                      <span key={tag} className="model-card-tag">
                        {tag}
                      </span>
                    ))}
                  </div>

                  <div className="model-card-footer">
                    <Link className="model-card-action" href={`/model-catalog/${model.id}`}>
                      View Details &amp; Code →
                    </Link>
                  </div>
                </div>
              </article>
            ))}
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
