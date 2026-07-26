"use client";

import Link from "next/link";
import { useState } from "react";
import { Download, MessagesSquare, Package, Play, Square } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatBytes, formatSeconds } from "@/features/platform/utils";
import { Badge, Button, EmptyState, Field, MutationError } from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import type { ModelExportFormat, ModelExportStatus, ModelInfo, ServingStatus } from "@/types/api";

// Which export formats each family can produce, mirroring the backend
// FORMATS_BY_FAMILY. An adapter can do everything; an already-merged HF model
// or a GGUF only produces (or re-quantizes to) GGUF.
const FORMATS_BY_FAMILY: Record<string, ModelExportFormat[]> = {
  llm_adapter: ["adapter_zip", "merged_16bit", "gguf_q4_k_m", "gguf_q5_k_m", "gguf_q8_0", "gguf_f16"],
  llm_hf: ["gguf_q4_k_m", "gguf_q5_k_m", "gguf_q8_0", "gguf_f16"],
  llm_gguf: ["gguf_q4_k_m", "gguf_q5_k_m", "gguf_q8_0", "gguf_f16"]
};

const FORMAT_LABEL: Record<ModelExportFormat, string> = {
  adapter_zip: "LoRA adapter (zip)",
  merged_16bit: "Merged 16-bit safetensors",
  gguf_q4_k_m: "GGUF Q4_K_M",
  gguf_q5_k_m: "GGUF Q5_K_M",
  gguf_q8_0: "GGUF Q8_0",
  gguf_f16: "GGUF F16"
};

const GGUF_QUANTS: ModelExportFormat[] = ["gguf_q4_k_m", "gguf_q5_k_m", "gguf_q8_0", "gguf_f16"];

// One-line "what you get" for the currently chosen format, shown under the
// chips so the choice is legible without reading the intro paragraph.
const FORMAT_DESC: Record<string, string> = {
  adapter_zip: "Just the tuned delta (smallest) — needs the base model to run.",
  merged_16bit: "A full Hugging Face checkpoint to download (largest).",
  gguf: "A standalone quantized model — registers a servable model you can chat with."
};

const EXPORT_ACTIVE = new Set(["queued", "running"]);

function exportIntro(family: string): string {
  if (family === "llm_adapter") {
    return (
      "Training saves a LoRA adapter — the small tuned delta over the base model, not a full model. " +
      "That is intentional: the adapter is a fraction of the size and stays faithful to the base. " +
      "Merge it into the base here to get a standalone artifact: GGUF to serve and chat, merged 16-bit " +
      "to download a full Hugging Face model, or adapter zip to keep just the delta."
    );
  }
  if (family === "llm_hf") {
    return "This is a full Hugging Face checkpoint. Export to GGUF to serve it in the chat runtime.";
  }
  return "This GGUF model is already servable. Re-quantize here to trade size for quality.";
}

function exportTone(status: string): "ok" | "fail" | "info" | "neutral" {
  if (status === "completed") return "ok";
  if (status === "failed") return "fail";
  if (EXPORT_ACTIVE.has(status)) return "info";
  return "neutral";
}

export function ExportPanel({ model }: { model: ModelInfo }) {
  const queryClient = useQueryClient();
  const families = FORMATS_BY_FAMILY[model.family] ?? [];
  const baseFormats = families.filter((format) => !format.startsWith("gguf_"));
  const canGguf = families.some((format) => format.startsWith("gguf_"));
  const [selected, setSelected] = useState<ModelExportFormat>(families[0] ?? "gguf_q4_k_m");
  const [ggufQuant, setGgufQuant] = useState<ModelExportFormat>(
    GGUF_QUANTS.find((format) => families.includes(format)) ?? "gguf_q4_k_m"
  );

  const exportsQuery = useQuery({
    queryKey: ["model-exports", model.id],
    queryFn: () => api.modelExports(model.id),
    refetchInterval: (query) =>
      (query.state.data ?? []).some((item) => EXPORT_ACTIVE.has(item.status)) ? 2500 : false
  });

  const startExport = useMutation({
    mutationFn: (format: ModelExportFormat) => api.createModelExport(model.id, format),
    onSuccess: async () => {
      toast.success("Export started");
      await queryClient.invalidateQueries({ queryKey: ["model-exports", model.id] });
    }
  });

  const isGgufSelected = selected.startsWith("gguf_");
  const effectiveFormat = isGgufSelected ? ggufQuant : selected;
  const activeExports = (exportsQuery.data ?? []).filter((item) => EXPORT_ACTIVE.has(item.status));
  const description = FORMAT_DESC[isGgufSelected ? "gguf" : selected] ?? "";

  return (
    <section className="panel export-panel">
      <div className="export-header">
        <h3 className="model-detail-section-title">Export</h3>
        <p className="model-detail-empty export-intro">{exportIntro(model.family)}</p>
      </div>

      <div className="export-config">
        <span className="export-label">Format</span>
        <div className="export-format-group" role="group" aria-label="Export format">
          {baseFormats.map((format) => (
            <FormatChip
              key={format}
              label={FORMAT_LABEL[format]}
              active={selected === format}
              onClick={() => setSelected(format)}
            />
          ))}
          {canGguf && (
            <FormatChip
              label="GGUF (quantized)"
              active={isGgufSelected}
              onClick={() => setSelected(ggufQuant)}
            />
          )}
        </div>
        {description ? <p className="export-format-desc">{description}</p> : null}

        <div className="export-controls">
          {isGgufSelected && (
            <Field label="Quantization">
              <select
                value={ggufQuant}
                onChange={(event) => {
                  const next = event.target.value as ModelExportFormat;
                  setGgufQuant(next);
                  setSelected(next);
                }}
              >
                {GGUF_QUANTS.filter((format) => families.includes(format)).map((format) => (
                  <option key={format} value={format}>
                    {FORMAT_LABEL[format]}
                  </option>
                ))}
              </select>
            </Field>
          )}
          <Button
            className="export-run-button"
            onClick={() => startExport.mutate(effectiveFormat)}
            disabled={startExport.isPending || !families.length}
          >
            <Package size={16} /> Export {FORMAT_LABEL[effectiveFormat]}
          </Button>
        </div>
        <MutationError mutations={[startExport]} />
      </div>

      <div className="export-history">
        <div className="export-history-head">
          <span className="export-label">Recent exports</span>
          {activeExports.length > 0 ? <Badge tone="info">{activeExports.length} running</Badge> : null}
        </div>
        <ExportList
          modelId={model.id}
          exports={exportsQuery.data ?? []}
          loading={exportsQuery.isLoading}
        />
      </div>
    </section>
  );
}

function FormatChip({
  label,
  active,
  onClick
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`export-format-chip${active ? " export-format-chip-active" : ""}`}
      aria-pressed={active}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function ExportList({
  modelId,
  exports,
  loading
}: {
  modelId: string;
  exports: ModelExportStatus[];
  loading: boolean;
}) {
  if (loading) return <p className="model-detail-empty">Loading exports…</p>;
  if (exports.length === 0) {
    return <p className="model-detail-empty">No exports yet. Choose a format above to create one.</p>;
  }
  return (
    <ul className="export-list">
      {exports.map((item) => (
        <li key={item.id} className="export-row">
          <div className="export-row-head">
            <div className="export-row-meta">
              <span className="export-row-format">{FORMAT_LABEL[item.format]}</span>
              <Badge tone={exportTone(item.status)}>{item.status}</Badge>
              {item.size_bytes ? (
                <span className="export-row-size">{formatBytes(item.size_bytes)}</span>
              ) : null}
            </div>
            {item.status === "completed" && item.artifact_name ? (
              <a
                className="secondary-button export-row-download"
                href={api.modelExportDownloadUrl(modelId, item.id)}
                download
              >
                <Download size={15} /> Download
              </a>
            ) : null}
          </div>
          {item.registered_model_id ? (
            <Link className="model-detail-link" href={`/models/${item.registered_model_id}`}>
              Servable GGUF model registered →
            </Link>
          ) : null}
          {EXPORT_ACTIVE.has(item.status) && item.logs.length > 0 ? (
            <div className="export-log">
              <span className="export-log-label">Live output</span>
              <pre className="export-log-tail" aria-live="polite">
                {item.logs.slice(-8).join("\n")}
              </pre>
            </div>
          ) : null}
          {item.status === "failed" && item.error ? (
            <p className="error-text export-row-error">{item.error}</p>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

function servingTone(state: string): "ok" | "info" | "fail" | "neutral" {
  if (state === "running") return "ok";
  if (state === "starting" || state === "stopping") return "info";
  return "neutral";
}

export function ServingPanel({ model }: { model: ModelInfo }) {
  const queryClient = useQueryClient();
  const statusQuery = useQuery({
    queryKey: ["serving-status"],
    queryFn: () => api.servingStatus(),
    refetchInterval: (query) => {
      const state = (query.state.data as ServingStatus | undefined)?.state;
      return state === "starting" || state === "stopping" ? 1500 : 4000;
    }
  });
  const status = statusQuery.data;
  const servingThis = status?.state === "running" && status.model_id === model.id;

  const startServing = useMutation({
    mutationKey: ["serving-start"],
    mutationFn: () => api.startServing({ model_id: model.id }),
    onSuccess: async (next) => {
      toast.success(`Serving ${next.model_name ?? model.name}`);
      await queryClient.invalidateQueries({ queryKey: ["serving-status"] });
    }
  });
  const stopServing = useMutation({
    mutationFn: () => api.stopServing(),
    onSuccess: async () => {
      toast.success("Serving stopped");
      await queryClient.invalidateQueries({ queryKey: ["serving-status"] });
    }
  });

  const busy = startServing.isPending || stopServing.isPending;
  const otherModelServing = status?.state === "running" && status.model_id !== model.id;

  return (
    <section className="panel">
      <h3 className="model-detail-section-title">Serving</h3>
      <div className="serving-status-row">
        <Badge tone={servingThis ? servingTone(status?.state ?? "stopped") : "neutral"}>
          {servingThis ? status?.state : "stopped"}
        </Badge>
        {servingThis && status?.port ? (
          <span className="serving-status-meta">
            port {status.port}
            {status.uptime_seconds != null ? ` · up ${formatSeconds(status.uptime_seconds)}` : ""}
          </span>
        ) : (
          <span className="serving-status-meta">This model is not being served.</span>
        )}
      </div>

      {otherModelServing ? (
        <p className="model-detail-empty">
          Another model is currently served ({status?.model_name}). Starting this one stops it —
          only one model serves at a time.
        </p>
      ) : null}

      <div className="serving-actions">
        {servingThis ? (
          <>
            <Button variant="secondary" onClick={() => stopServing.mutate()} disabled={busy}>
              <Square size={16} /> Stop serving
            </Button>
            <Link className="secondary-button" href="/inference/chat">
              <MessagesSquare size={16} /> Open chat
            </Link>
          </>
        ) : (
          <Button onClick={() => startServing.mutate()} disabled={busy}>
            <Play size={16} /> {status?.state === "starting" ? "Starting…" : "Start serving"}
          </Button>
        )}
      </div>

      {status?.state === "stopped" && status.error ? (
        <div className="serving-error">
          <p className="error-text">{status.error}</p>
          {status.stderr_tail.length > 0 ? (
            <pre className="export-log-tail">{status.stderr_tail.slice(-8).join("\n")}</pre>
          ) : null}
        </div>
      ) : null}

      <MutationError mutations={[startServing, stopServing]} />
    </section>
  );
}

export function ExportToGgufCta() {
  return (
    <section className="panel model-gate-panel">
      <h3 className="model-detail-section-title">Serving</h3>
      <EmptyState
        icon={<MessagesSquare size={28} />}
        label="Export to GGUF first"
        description="Only GGUF models run on the managed llama.cpp chat runtime. Use the Export panel above to produce a GGUF, then serve the registered GGUF model."
      />
    </section>
  );
}
