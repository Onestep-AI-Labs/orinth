"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { SYSTEM_MODELS, type SystemModel } from "./models-data";

export interface FeaturedModelCard {
  model: SystemModel;
  categoryLabel: string;
  highlightMetric: { label: string; value: string };
}

export const FEATURED_MODELS: FeaturedModelCard[] = [
  {
    model: SYSTEM_MODELS.find((m) => m.id === "yolo_11_best")!,
    categoryLabel: "Computer Vision",
    highlightMetric: { label: "mAP@50", value: "0.892" }
  },
  {
    model: SYSTEM_MODELS.find((m) => m.id === "llm_gguf_chat")!,
    categoryLabel: "LLM & Serving",
    highlightMetric: { label: "Speed", value: "34.2 t/s" }
  },
  {
    model: SYSTEM_MODELS.find((m) => m.id === "unet_inception")!,
    categoryLabel: "Deep Vision",
    highlightMetric: { label: "Accuracy", value: "94.2%" }
  },
  {
    model: SYSTEM_MODELS.find((m) => m.id === "hf_transformers_nlp")!,
    categoryLabel: "NLP Transformers",
    highlightMetric: { label: "F1 Score", value: "0.941" }
  },
  {
    model: SYSTEM_MODELS.find((m) => m.id === "keyword_text_classifier")!,
    categoryLabel: "NLP Baseline",
    highlightMetric: { label: "Latency", value: "<1 ms" }
  },
  {
    model: SYSTEM_MODELS.find((m) => m.id === "extractive_summarizer")!,
    categoryLabel: "NLP Baseline",
    highlightMetric: { label: "ROUGE-1", value: "42.1" }
  },
  {
    model: SYSTEM_MODELS.find((m) => m.id === "keyword_qa")!,
    categoryLabel: "NLP Retrieval",
    highlightMetric: { label: "Exact Match", value: "78.4%" }
  },
  {
    model: SYSTEM_MODELS.find((m) => m.id === "architecture_studio_custom")!,
    categoryLabel: "Architecture Studio",
    highlightMetric: { label: "Presets", value: "25+ Layers" }
  }
];

export function PopularModelsSlider() {
  // Page index (0 or 1 for 4 items per page, out of 8 total items)
  const [currentPage, setCurrentPage] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const totalPages = 2; // 8 items total, 4 per slide page

  const nextPage = useCallback(() => {
    setCurrentPage((prev) => (prev + 1) % totalPages);
  }, [totalPages]);

  const prevPage = useCallback(() => {
    setCurrentPage((prev) => (prev - 1 + totalPages) % totalPages);
  }, [totalPages]);

  // 5-second automatic sliding to the right
  useEffect(() => {
    if (isPaused) return;

    timerRef.current = setInterval(() => {
      nextPage();
    }, 5000);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isPaused, nextPage]);

  return (
    <div
      className="popular-slider-wrapper"
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      <div className="popular-slider-controls-top">
        <div className="popular-slider-dots">
          {Array.from({ length: totalPages }).map((_, idx) => (
            <button
              key={idx}
              className={`popular-slider-dot ${idx === currentPage ? "active" : ""}`}
              onClick={() => setCurrentPage(idx)}
              aria-label={`Go to page ${idx + 1}`}
            />
          ))}
        </div>

        <div className="popular-slider-nav-btns">
          <button
            className="popular-slider-nav-btn"
            onClick={prevPage}
            aria-label="Previous models page"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M15 18l-6-6 6-6" />
            </svg>
          </button>
          <button
            className="popular-slider-nav-btn"
            onClick={nextPage}
            aria-label="Next models page"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 18l6-6-6-6" />
            </svg>
          </button>
        </div>
      </div>

      <div className="popular-slider-track-container">
        <div
          className="popular-slider-track"
          style={{
            transform: `translateX(-${currentPage * 100}%)`,
            transition: "transform 600ms cubic-bezier(0.16, 1, 0.3, 1)"
          }}
        >
          {/* Page 1 (Items 0..3) */}
          <div className="popular-slider-page">
            <div className="popular-models-grid-4">
              {FEATURED_MODELS.slice(0, 4).map((item) => (
                <SimpleModelCard key={item.model.id} item={item} />
              ))}
            </div>
          </div>

          {/* Page 2 (Items 4..7) */}
          <div className="popular-slider-page">
            <div className="popular-models-grid-4">
              {FEATURED_MODELS.slice(4, 8).map((item) => (
                <SimpleModelCard key={item.model.id} item={item} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SimpleModelCard({ item }: { item: FeaturedModelCard }) {
  const { model, categoryLabel, highlightMetric } = item;

  return (
    <article className="simple-model-card">
      <div className="simple-model-card-header">
        <span className="simple-model-category-tag">{categoryLabel}</span>
        <span className="simple-model-format-pill">{model.format}</span>
      </div>

      <div className="simple-model-card-main">
        <h3 className="simple-model-title">{model.name}</h3>
        <span className="simple-model-family">{model.family}</span>
        <p className="simple-model-desc">{model.description}</p>
      </div>

      <div className="simple-model-card-footer">
        <div className="simple-model-metric">
          <span className="simple-model-metric-val">{highlightMetric.value}</span>
          <span className="simple-model-metric-lbl">{highlightMetric.label}</span>
        </div>

        <Link href={`/model-catalog/${model.id}`} className="simple-model-action">
          View Specs
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </Link>
      </div>
    </article>
  );
}
