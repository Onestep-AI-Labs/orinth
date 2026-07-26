"use client";

import Link from "next/link";
import { useState } from "react";
import { AlertTriangle, ArrowRight, ExternalLink, Play, Sparkles, X } from "lucide-react";
import { Badge, Field, InlineSpinner, MutationError, NumberInput, PanelTitle } from "@/features/platform/ui";
import {
  useCancelRecipeMutation,
  useGenerateRecipeMutation,
  useOpenRouterModelsQuery,
  usePlatformSettingsQuery
} from "@/features/recipes/hooks";
import type { RecipeGenerationMode, RecipePromptFlavor, RecipeRead } from "@/types/api";

const MODES: { value: RecipeGenerationMode; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "LLM when a key is configured, rules otherwise" },
  { value: "llm", label: "LLM", hint: "Requires an OpenRouter key" },
  { value: "rules", label: "Rules", hint: "Deterministic scaffolding, no LLM" }
];

const FLAVORS: RecipePromptFlavor[] = ["qa", "instruction", "conversation"];

export function RecipeGenerateStep({ recipe, onReview }: { recipe: RecipeRead; onReview: () => void }) {
  const settingsQuery = usePlatformSettingsQuery();
  const modelsQuery = useOpenRouterModelsQuery();
  const generateMutation = useGenerateRecipeMutation(recipe.id);
  const cancelMutation = useCancelRecipeMutation(recipe.id);

  const [mode, setMode] = useState<RecipeGenerationMode>(recipe.generation.mode ?? "auto");
  const [flavor, setFlavor] = useState<RecipePromptFlavor>(recipe.generation.prompt_flavor ?? "qa");
  const [chunkSize, setChunkSize] = useState(recipe.generation.chunk_size ?? 3000);
  const [chunkOverlap, setChunkOverlap] = useState(recipe.generation.chunk_overlap ?? 200);
  const [recordsPerChunk, setRecordsPerChunk] = useState(recipe.generation.records_per_chunk ?? 3);
  const [model, setModel] = useState(recipe.generation.model ?? "");

  const keyConfigured = Boolean(settingsQuery.data?.openrouter_api_key_configured);
  const platformModel = settingsQuery.data?.openrouter_model ?? null;
  const busy = recipe.status === "generating" || recipe.status === "extracting";
  const models = modelsQuery.data?.models ?? [];

  function runGenerate() {
    generateMutation.mutate({
      mode,
      prompt_flavor: flavor,
      chunk_size: chunkSize,
      chunk_overlap: chunkOverlap,
      records_per_chunk: recordsPerChunk,
      model: model || null
    });
  }

  return (
    <section className="panel">
      <PanelTitle icon={<Sparkles size={18} />} title="Generate records" />

      <div className={`recipe-openrouter-banner ${keyConfigured ? "recipe-openrouter-on" : "recipe-openrouter-off"}`}>
        <span>
          <Badge tone={keyConfigured ? "ok" : "neutral"}>{keyConfigured ? "OpenRouter connected" : "OpenRouter not configured"}</Badge>
          {keyConfigured
            ? platformModel
              ? ` Default model: ${platformModel}`
              : " No default model selected"
            : " Runs fall back to deterministic rule-based records."}
        </span>
        <Link className="recipe-openrouter-link" href="/settings">
          Settings <ExternalLink size={13} />
        </Link>
      </div>

      <div className="recipe-generate-grid mt-4">
        <Field label="Mode" hint={MODES.find((entry) => entry.value === mode)?.hint}>
          <select value={mode} onChange={(event) => setMode(event.target.value as RecipeGenerationMode)}>
            {MODES.map((entry) => (
              <option key={entry.value} value={entry.value}>{entry.label}</option>
            ))}
          </select>
        </Field>
        <Field label="Prompt style">
          <select value={flavor} onChange={(event) => setFlavor(event.target.value as RecipePromptFlavor)}>
            {FLAVORS.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </Field>
        <Field label="Model override">
          <select value={model} onChange={(event) => setModel(event.target.value)} disabled={mode === "rules"}>
            <option value="">{platformModel ? `Default (${platformModel})` : "Platform default"}</option>
            {models.map((entry) => (
              <option key={entry.id} value={entry.id}>{entry.name}</option>
            ))}
          </select>
        </Field>
        <Field label="Chunk size (chars)">
          <NumberInput value={chunkSize} min={200} max={20000} step={100} onChange={setChunkSize} />
        </Field>
        <Field label="Chunk overlap">
          <NumberInput value={chunkOverlap} min={0} max={4000} step={50} onChange={setChunkOverlap} />
        </Field>
        <Field label="Records per chunk">
          <NumberInput value={recordsPerChunk} min={1} max={20} step={1} onChange={setRecordsPerChunk} />
        </Field>
      </div>

      {mode === "llm" && !keyConfigured && (
        <p className="error-text mt-3">
          LLM mode needs an OpenRouter API key. Add one in Settings, or switch to Auto or Rules.
        </p>
      )}

      {recipe.warnings.length > 0 && (
        <div className="recipe-warning-panel mt-4">
          <span className="recipe-warning-panel-head">
            <AlertTriangle size={15} /> Run warnings
          </span>
          <ul>
            {recipe.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {recipe.error && <p className="error-text mt-3">{recipe.error}</p>}

      <MutationError mutations={[generateMutation, cancelMutation]} />

      <div className="recipe-step-footer">
        <span className="text-sm text-ink-subtle">
          {busy ? "Generating…" : `${recipe.record_count} records generated`}
        </span>
        <div className="flex items-center gap-2">
          {busy && <InlineSpinner label="Working" />}
          {busy ? (
            <button className="secondary-button" onClick={() => cancelMutation.mutate()} disabled={cancelMutation.isPending}>
              <X size={16} /> Cancel
            </button>
          ) : (
            <button
              className="primary-button"
              onClick={runGenerate}
              disabled={generateMutation.isPending || (mode === "llm" && !keyConfigured)}
            >
              <Play size={16} /> {recipe.record_count > 0 ? "Regenerate" : "Generate"}
            </button>
          )}
          {recipe.record_count > 0 && !busy && (
            <button className="primary-button" onClick={onReview}>
              Review <ArrowRight size={16} />
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
