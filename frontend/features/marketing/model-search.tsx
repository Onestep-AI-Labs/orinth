"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { SYSTEM_MODELS, type SystemModel } from "./models-data";

export function ModelSearch() {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [results, setResults] = useState<SystemModel[]>([]);
  const searchRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    if (!query.trim()) {
      setResults(SYSTEM_MODELS.slice(0, 4));
      return;
    }

    const q = query.toLowerCase();
    const filtered = SYSTEM_MODELS.filter((model) => {
      const matchName = model.name.toLowerCase().includes(q);
      const matchFamily = model.family.toLowerCase().includes(q);
      const matchLabel = model.taskLabel.toLowerCase().includes(q);
      const matchDesc = model.description.toLowerCase().includes(q);
      const matchTags = model.tags.some((tag) => tag.toLowerCase().includes(q));
      return matchName || matchFamily || matchLabel || matchDesc || matchTags;
    });

    setResults(filtered);
  }, [query]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (modelId: string) => {
    setIsOpen(false);
    setQuery("");
    router.push(`/model-catalog/${modelId}`);
  };

  return (
    <div className="landing-search" ref={searchRef}>
      <div className="landing-search-input-wrap">
        <svg
          className="landing-search-icon"
          width="15"
          height="15"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>

        <input
          type="text"
          className="landing-search-input"
          placeholder="Search models (e.g. YOLO, U-Net, LLM, BERT)..."
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              setIsOpen(false);
            } else if (e.key === "Enter" && results.length > 0) {
              handleSelect(results[0].id);
            }
          }}
        />

        {query && (
          <button
            type="button"
            className="landing-search-clear"
            onClick={() => {
              setQuery("");
              setResults(SYSTEM_MODELS.slice(0, 4));
            }}
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {isOpen && (
        <div className="landing-search-dropdown">
          <div className="landing-search-dropdown-header">
            {query.trim() ? (
              <span>
                Matching models ({results.length})
              </span>
            ) : (
              <span>Popular Available Models</span>
            )}
          </div>

          {results.length === 0 ? (
            <div className="landing-search-empty">
              No models found matching &quot;{query}&quot;. Try searching for <em>YOLO</em>, <em>LLM</em>, or <em>Segmentation</em>.
            </div>
          ) : (
            <ul className="landing-search-list">
              {results.map((model) => (
                <li key={model.id}>
                  <button
                    type="button"
                    className="landing-search-item"
                    onClick={() => handleSelect(model.id)}
                  >
                    <div className="landing-search-item-header">
                      <strong className="landing-search-item-title">{model.name}</strong>
                      <span className="landing-search-item-tag">{model.taskLabel}</span>
                    </div>
                    <p className="landing-search-item-desc">{model.description}</p>
                    <div className="landing-search-item-meta">
                      <span>Family: {model.family}</span>
                      <span>Format: {model.format}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="landing-search-dropdown-footer">
            <button
              type="button"
              className="landing-search-all-link"
              onClick={() => {
                setIsOpen(false);
                router.push("/model-catalog");
              }}
            >
              Browse all {SYSTEM_MODELS.length} models in catalog →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
