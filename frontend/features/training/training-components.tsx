"use client";

/* eslint-disable @next/next/no-img-element */

import Link from "next/link";
import { mediaUrl } from "@/lib/api";
import {
  formatDatasetTask,
  formatMetric,
  formatSeconds,
  isActiveStatus,
  isLlmTask,
  isNlpTask,
  labelColor,
  normalizeTerminalOutput
} from "@/features/platform/utils";
import { dedupeConsecutive, Metric, StatusBadge, toggleId } from "@/features/platform/ui";
import { type ChartSeries, TrainingChart } from "@/features/training/training-chart";
import type { TaskType, TrainingJob } from "@/types/api";

export function TrainingJobTable({
  jobs,
  selectedIds,
  setSelectedIds
}: {
  jobs: TrainingJob[];
  selectedIds: string[];
  setSelectedIds: (ids: string[]) => void;
}) {
  return (
    <div className="table-wrap responsive-card-table">
      <table>
        <thead>
          <tr>
            <th />
            <th>Status</th>
            <th>Family</th>
            <th>Dataset</th>
            <th>Epoch</th>
            <th>Metric</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td data-label="Select">
                <input
                  type="checkbox"
                  checked={selectedIds.includes(job.id)}
                  onChange={(event) => toggleId(job.id, event.target.checked, selectedIds, setSelectedIds)}
                />
              </td>
              <td data-label="Status"><StatusBadge status={job.status} /></td>
              <td data-label="Family">{job.model_family}</td>
              <td data-label="Dataset">{job.parameters?.dataset_id ?? "-"}</td>
              <td data-label="Epoch">{(job.metrics as Record<string, any> | undefined)?.epoch ?? job.progress.processed ?? "-"}</td>
              <td data-label="Metric">{formatMetric(job.metrics?.["metrics/mAP50(B)"] ?? job.metrics?.val_accuracy ?? job.metrics?.accuracy ?? job.metrics?.macro_f1 ?? job.metrics?.rougeL ?? job.metrics?.f1 ?? job.metrics?.val_loss ?? job.metrics?.loss)}</td>
              <td data-label="Open"><Link className="secondary-button" href={`/training/${job.id}`}>Details</Link></td>
            </tr>
          ))}
          {jobs.length === 0 && (
            <tr>
              <td className="table-empty-cell" colSpan={7}>No training jobs</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export function TrainingDetails({ job }: { job: TrainingJob }) {
  const active = isActiveStatus(job.status);
  const liveMetrics = liveMetricRows(job);
  const groups = chartGroups(job);
  const advancedRows = advancedTrainingParameters(job);
  // Normalize before deduping: the runner streams raw TTY output, and two
  // progress-bar frames that differ only in cursor control bytes are the same
  // line once rendered.
  const logLines = dedupeConsecutive(
    job.progress.logs.map(normalizeTerminalOutput).filter((line) => line.length > 0)
  ).slice(-80);
  return (
    <div className="training-detail-body">
      {liveMetrics.length > 0 && (
        <section className="training-section">
          <h3 className="section-title">{active ? "Live metrics" : "Results"}</h3>
          <div className="metric-grid">
            {liveMetrics.map((metric) => (
              <Metric key={metric.label} label={metric.label} value={metric.value} />
            ))}
          </div>
        </section>
      )}
      {groups.length > 0 && (
        <section className="training-section">
          <h3 className="section-title">Training graphs</h3>
          <div className="training-chart-grid">
            {groups.map((group, index) => (
              <TrainingChart
                key={group.title}
                title={group.title}
                rows={job.history}
                series={group.series}
                xKey={group.xKey}
                xLabel={group.xLabel}
                colorOffset={index}
              />
            ))}
          </div>
        </section>
      )}
      <SampleGenerations job={job} />
      <TrainingRocPanel curves={job.curves} />
      <TrainingArtifactImages urls={job.artifact_urls ?? {}} />
      <section className="training-section training-params-grid">
        <KeyValueTable title="Parameters" value={visibleTrainingParameters(job)} />
        {Object.keys(advancedRows).length > 0 && (
          <KeyValueTable title="Advanced settings" value={advancedRows} />
        )}
      </section>
      {logLines.length > 0 && (
        <section className="training-section">
          <h3 className="section-title">Run log</h3>
          <div className="log-box">
            {logLines.map((line, index) => (
              <pre key={`${line}-${index}`}>{line}</pre>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

type LiveMetric = { label: string; value: string };

// Curated live readout: the values that actually change during a run, ordered
// by relevance and filtered to those a given family reports. Covers LLM
// (loss/val loss/perplexity/lr), detection (mAP), and classification/NLP
// (accuracy/F1/ROUGE) from one list so the panel adapts to the model family.
function liveMetricRows(job: TrainingJob): LiveMetric[] {
  const progress = job.progress;
  const metrics = (job.metrics ?? {}) as Record<string, unknown>;
  const history = job.history ?? [];
  const num = (value: unknown): number | undefined =>
    typeof value === "number" && Number.isFinite(value) ? value : undefined;
  const best = (key: string, mode: "min" | "max"): number | undefined => {
    const values = history.map((row) => Number(row[key])).filter((value) => Number.isFinite(value));
    if (values.length === 0) return undefined;
    return mode === "min" ? Math.min(...values) : Math.max(...values);
  };
  const fmtLr = (value: number) => (value < 0.01 ? value.toExponential(1) : value.toFixed(4));
  const count = progress.total
    ? `${progress.processed}/${progress.total}`
    : progress.processed
      ? String(progress.processed)
      : undefined;

  const candidates: Array<{ label: string; value?: string } | null> = [
    { label: "Progress", value: `${Math.round(progress.percent)}%` },
    { label: "Elapsed", value: formatSeconds(progress.elapsed_seconds) },
    progress.eta_seconds ? { label: "ETA", value: formatSeconds(progress.eta_seconds) } : null,
    count ? { label: isLlmTask(String(job.task_type)) ? "Step" : "Epoch", value: count } : null,
    { label: "Loss", value: num(metrics.loss)?.toFixed(4) },
    { label: "Val loss", value: num(metrics.val_loss)?.toFixed(4) },
    { label: "Best val loss", value: best("val_loss", "min")?.toFixed(4) },
    { label: "Train loss", value: num(metrics.train_loss)?.toFixed(4) },
    { label: "Perplexity", value: num(metrics.perplexity)?.toFixed(2) },
    { label: "Learning rate", value: (() => { const lr = num(metrics.learning_rate) ?? num(metrics.lr); return lr === undefined ? undefined : fmtLr(lr); })() },
    { label: "Accuracy", value: (num(metrics.val_accuracy) ?? num(metrics.accuracy))?.toFixed(4) },
    { label: "mAP@50", value: num(metrics["metrics/mAP50(B)"])?.toFixed(4) },
    { label: "Macro F1", value: (num(metrics.macro_f1) ?? num(metrics.f1))?.toFixed(4) },
    { label: "ROUGE-L", value: num(metrics.rougeL)?.toFixed(4) },
    { label: "Train records", value: num(metrics.train_records)?.toString() }
  ];
  return candidates
    .filter((entry): entry is LiveMetric => entry !== null && Boolean(entry.value))
    .slice(0, 12);
}

type ChartGroup = { title: string; series: ChartSeries[]; xKey?: string; xLabel: string };

// Panels grouped by a shared Y scale so overlaid series stay comparable — the
// reference "Model Performance" chart puts mAP curves together and losses in
// their own panels. Only groups with at least one available column render.
const CHART_GROUP_TEMPLATES: Array<{ title: string; series: ChartSeries[] }> = [
  { title: "Loss", series: [{ key: "loss", label: "Train loss" }, { key: "val_loss", label: "Validation loss" }] },
  { title: "Model performance (mAP)", series: [{ key: "metrics/mAP50(B)", label: "mAP@50" }, { key: "metrics/mAP50-95(B)", label: "mAP@50-95" }] },
  { title: "Precision / recall", series: [{ key: "metrics/precision(B)", label: "Precision" }, { key: "metrics/recall(B)", label: "Recall" }] },
  { title: "Detection losses", series: [{ key: "train/box_loss", label: "Box" }, { key: "train/cls_loss", label: "Class" }, { key: "train/dfl_loss", label: "DFL" }] },
  { title: "Accuracy", series: [{ key: "accuracy", label: "Accuracy" }, { key: "val_accuracy", label: "Validation accuracy" }, { key: "train_accuracy", label: "Train accuracy" }, { key: "eval_accuracy", label: "Eval accuracy" }, { key: "metrics/accuracy", label: "Accuracy" }] },
  { title: "Quality scores", series: [{ key: "macro_f1", label: "Macro F1" }, { key: "f1", label: "F1" }, { key: "rougeL", label: "ROUGE-L" }] }
];

function chartGroups(job: TrainingJob): ChartGroup[] {
  const history = job.history ?? [];
  if (history.length < 2) return [];
  const available = new Set<string>();
  for (const row of history) {
    for (const [key, value] of Object.entries(row)) {
      if (Number.isFinite(Number(value))) available.add(key);
    }
  }
  const isLlm = isLlmTask(String(job.task_type));
  const xKey = available.has("step") ? "step" : available.has("epoch") ? "epoch" : undefined;
  const xLabel = xKey === "step" || (isLlm && !xKey) ? "Step" : "Epoch";
  const groups: ChartGroup[] = [];
  for (const template of CHART_GROUP_TEMPLATES) {
    const series = template.series.filter((entry) => available.has(entry.key));
    if (series.length > 0) groups.push({ title: template.title, series, xKey, xLabel });
  }
  return groups;
}

function visibleTrainingParameters(job: TrainingJob): Record<string, any> {
  const params = job.parameters ?? {};
  const hyperparameters = params.hyperparameters ?? {};
  const taskType = String(params.task_type ?? job.task_type ?? "");
  const modelOptionId = String(params.model_option_id ?? "");
  const rows: Record<string, any> = {
    Task: formatDatasetTask(taskType as TaskType),
    Model: modelOptionId || job.model_family,
    Dataset: params.dataset_id ?? "-"
  };
  if (params.model_name) rows["Model name"] = params.model_name;
  if (params.epochs !== undefined) rows.Epochs = params.epochs;

  if (isLlmTask(taskType)) {
    rows.Method = LLM_METHOD_LABELS[String(hyperparameters.finetune_method ?? "lora")] ?? "LoRA adapter";
    rows.Epochs = params.epochs ?? "-";
    rows.Batch = params.batch_size ?? "-";
    rows.LR = params.learning_rate ?? "-";
    return rows;
  }

  if (isNlpTask(taskType)) {
    if (modelOptionId === "nlp_tfidf_classifier") {
      rows.Strategy = "TF-IDF + Logistic Regression";
      rows.C = params.learning_rate ?? "-";
      return rows;
    }
    if (modelOptionId === "nlp_extractive_summarizer") {
      rows.Strategy = "Extractive summary baseline";
      return rows;
    }
    if (modelOptionId === "nlp_keyword_qa") {
      rows.Strategy = "Keyword QA baseline";
      return rows;
    }
    rows.Batch = params.batch_size ?? "-";
    rows.Optimizer = params.optimizer ?? "-";
    rows.LR = params.learning_rate ?? "-";
    if (hyperparameters.max_length !== undefined) rows["Max length"] = hyperparameters.max_length;
    if (hyperparameters.target_max_length !== undefined) rows["Target length"] = hyperparameters.target_max_length;
    if (hyperparameters.vocab_size !== undefined) rows.Vocab = hyperparameters.vocab_size;
    return rows;
  }

  rows.Size = params.image_size ?? "-";
  rows.Batch = params.batch_size ?? "-";
  rows.Optimizer = params.optimizer ?? "-";
  rows.LR = params.learning_rate ?? "-";
  rows.Device = params.device || "auto";
  rows.Cache = params.cache ?? "-";
  rows.Workers = params.workers ?? "-";
  rows.Patience = params.patience ?? "-";
  return rows;
}

const LLM_METHOD_LABELS: Record<string, string> = {
  lora: "LoRA adapter",
  qlora: "QLoRA (4-bit) adapter",
  full: "Full fine-tune",
  continued_pretrain: "Continued pretraining (LoRA)"
};

// The NLP knobs `visibleTrainingParameters` already lists as basic fields; the
// rest of `hyperparameters` are advanced values, rendered so a finished run
// documents its own configuration.
const BASIC_HYPERPARAMETER_KEYS = new Set(["max_length", "target_max_length", "vocab_size", "finetune_method"]);

function advancedTrainingParameters(job: TrainingJob): Record<string, any> {
  const hyperparameters = (job.parameters?.hyperparameters ?? {}) as Record<string, any>;
  const rows: Record<string, any> = {};
  for (const [key, value] of Object.entries(hyperparameters)) {
    if (BASIC_HYPERPARAMETER_KEYS.has(key) || value === null || value === undefined) continue;
    rows[key.replaceAll("_", " ")] = typeof value === "boolean" ? (value ? "on" : "off") : value;
  }
  return rows;
}

type SampleGeneration = { prompt?: string; reference?: string; generated?: string };

/**
 * Qualitative outputs from the just-trained adapter (phase 14), rendered next
 * to the loss curves so a finished LLM run shows what it actually learned.
 */
function SampleGenerations({ job }: { job: TrainingJob }) {
  const generations = (job.artifacts?.sample_generations as SampleGeneration[] | undefined) ?? [];
  if (!Array.isArray(generations) || generations.length === 0) return null;
  return (
    <div>
      <h3 className="section-title">Sample Generations</h3>
      <div className="generation-grid">
        {generations.map((item, index) => (
          <div className="generation-card" key={`${item.prompt ?? ""}-${index}`}>
            <p className="generation-label">Prompt</p>
            <pre>{item.prompt || "—"}</pre>
            <p className="generation-label">Generated</p>
            <pre>{item.generated || "(no output)"}</pre>
            {item.reference ? (
              <>
                <p className="generation-label">Reference</p>
                <pre>{item.reference}</pre>
              </>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

function TrainingRocPanel({ curves }: { curves: Record<string, any> }) {
  const roc = curves?.roc as Record<string, { fpr: number[]; tpr: number[] }> | undefined;
  if (!roc || Object.keys(roc).length === 0) return null;
  return (
    <div className="chart-card">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <strong>ROC / AUC</strong>
        <span>macro {formatMetric(curves.macro_auc)} / micro {formatMetric(curves.micro_auc)}</span>
      </div>
      <svg className="roc-chart" viewBox="0 0 100 64" preserveAspectRatio="none">
        <line x1="0" y1="64" x2="100" y2="0" style={{ stroke: "var(--color-line)" }} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        {Object.entries(roc).map(([label, curve], index) => {
          const points = (curve.fpr ?? [])
            .map((fpr, pointIndex) => `${(fpr * 100).toFixed(2)},${(64 - ((curve.tpr?.[pointIndex] ?? 0) * 64)).toFixed(2)}`)
            .join(" ");
          return <polyline key={label} points={points} fill="none" style={{ stroke: labelColor(index) }} strokeWidth="2" vectorEffect="non-scaling-stroke" />;
        })}
      </svg>
      <div className="label-chip-row mt-3">
        {Object.keys(roc).map((label, index) => (
          <span className="label-chip" key={label}>
            <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

function TrainingArtifactImages({ urls }: { urls: Record<string, string> }) {
  const entries = Object.entries(urls).slice(0, 8);
  if (entries.length === 0) return null;
  return (
    <div>
      <h3 className="section-title">Curve Artifacts</h3>
      <div className="artifact-grid">
        {entries.map(([name, url]) => (
          <div className="artifact-card" key={name}>
            <strong>{name.replaceAll("_", " ")}</strong>
            <img src={mediaUrl(url) ?? url} alt={name} />
          </div>
        ))}
      </div>
    </div>
  );
}

function KeyValueTable({ title, value }: { title: string; value: Record<string, any> }) {
  return (
    <div>
      <h3 className="section-title">{title}</h3>
      <div className="table-wrap">
        <table>
          <tbody>
            {Object.entries(value).map(([key, item]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{typeof item === "object" ? JSON.stringify(item) : String(item)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
