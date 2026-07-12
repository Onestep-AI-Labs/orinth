"use client";

/* eslint-disable @next/next/no-img-element */

import Link from "next/link";
import { mediaUrl } from "@/lib/api";
import { formatMetric, labelColor } from "@/features/platform/utils";
import { Metric, StatusBadge, toggleId } from "@/features/platform/ui";
import type { TrainingJob } from "@/types/api";

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
              <td data-label="Epoch">{job.metrics?.epoch ?? job.progress.processed ?? "-"}</td>
              <td data-label="Metric">{formatMetric(job.metrics?.["metrics/mAP50(B)"] ?? job.metrics?.val_accuracy ?? job.metrics?.accuracy ?? job.metrics?.macro_f1 ?? job.metrics?.rougeL ?? job.metrics?.f1)}</td>
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
  const metricRows = Object.entries(job.metrics ?? {}).filter(([, value]) => typeof value !== "object");
  const chartKeys = trainingChartKeys(job);
  return (
    <div className="space-y-4">
      <KeyValueTable title="Parameters" value={job.parameters} />
      {metricRows.length > 0 && (
        <div className="metric-grid">
          {metricRows.slice(0, 10).map(([key, value]) => (
            <Metric key={key} label={key.replaceAll("_", " ")} value={formatMetric(value)} />
          ))}
        </div>
      )}
      {chartKeys.length > 0 && (
        <div className="chart-grid">
          {chartKeys.map((key, index) => (
            <MiniLineChart key={key} rows={job.history} valueKey={key} color={labelColor(index)} />
          ))}
        </div>
      )}
      <TrainingRocPanel curves={job.curves} />
      <TrainingArtifactImages urls={job.artifact_urls ?? {}} />
      {job.progress.logs.length > 0 && (
        <div className="log-box">
          {job.progress.logs.slice(-60).map((line, index) => (
            <pre key={`${line}-${index}`}>{line}</pre>
          ))}
        </div>
      )}
    </div>
  );
}

function trainingChartKeys(job: TrainingJob): string[] {
  if (!job.history?.length) return [];
  const preferred = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "train/box_loss",
    "train/cls_loss",
    "val_accuracy",
    "accuracy",
    "macro_f1",
    "rougeL",
    "f1",
    "val_loss",
    "loss"
  ];
  const available = new Set(Object.keys(job.history[0] ?? {}));
  return preferred.filter((key) => available.has(key)).slice(0, 6);
}

function MiniLineChart({ rows, valueKey, color }: { rows: Array<Record<string, any>>; valueKey: string; color: string }) {
  const values = rows.map((row) => Number(row[valueKey])).filter((value) => Number.isFinite(value));
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 0.000001);
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100;
      const y = 44 - ((value - min) / span) * 38;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <div className="chart-card">
      <div className="flex items-center justify-between gap-3">
        <strong title={valueKey}>{valueKey}</strong>
        <span>{formatMetric(values[values.length - 1])}</span>
      </div>
      <svg viewBox="0 0 100 48" preserveAspectRatio="none">
        <polyline points={points} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
      </svg>
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
        <line x1="0" y1="64" x2="100" y2="0" stroke="#d7dde5" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        {Object.entries(roc).map(([label, curve], index) => {
          const points = (curve.fpr ?? [])
            .map((fpr, pointIndex) => `${(fpr * 100).toFixed(2)},${(64 - ((curve.tpr?.[pointIndex] ?? 0) * 64)).toFixed(2)}`)
            .join(" ");
          return <polyline key={label} points={points} fill="none" stroke={labelColor(index)} strokeWidth="2" vectorEffect="non-scaling-stroke" />;
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
