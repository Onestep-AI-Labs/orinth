"use client";

import Link from "next/link";
import { FlaskConical } from "lucide-react";
import { areTasksCompatible, displayModelName, formatMetric } from "@/features/platform/utils";
import { EmptyState, Metric, StatusBadge, toggleId } from "@/features/platform/ui";
import type { EvaluationJob, EvaluationPerImageRow, TaskType } from "@/types/api";

type EvaluationDisplayKind = "classification" | "vision" | "text_classification" | "summarization" | "question_answering" | "llm";
type MetricEntry = { key: string; label: string; value: number };

export function TestingJobTable({
  jobs,
  selectedIds,
  setSelectedIds,
  modelTaskById = {},
  modelNameById = {},
  taskType
}: {
  jobs: EvaluationJob[];
  selectedIds: string[];
  setSelectedIds: (ids: string[]) => void;
  modelTaskById?: Record<string, TaskType>;
  modelNameById?: Record<string, string>;
  taskType?: TaskType;
}) {
  const metricColumns = testingTableColumns(jobs, modelTaskById, taskType);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th />
            <th>Status</th>
            <th>Model</th>
            <th>Dataset</th>
            <th>Samples</th>
            {metricColumns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td>
                <input
                  type="checkbox"
                  checked={selectedIds.includes(job.id)}
                  onChange={(event) => toggleId(job.id, event.target.checked, selectedIds, setSelectedIds)}
                />
              </td>
              <td data-label="Status"><StatusBadge status={job.status} /></td>
              <td data-label="Model">{displayModelName(job.model_id, modelNameById)}</td>
              <td data-label="Dataset">{job.dataset_key}</td>
              <td data-label="Samples">{metricsOf(job).samples ?? "-"}</td>
              {metricColumns.map((column) => (
                <td data-label={column.label} key={column.key}>{formatMetric(column.get(job))}</td>
              ))}
              <td data-label="Open"><Link className="secondary-button" href={`/testing/${job.id}`}>Details</Link></td>
            </tr>
          ))}
          {jobs.length === 0 && (
            <tr>
              <td className="table-empty-cell" colSpan={6 + metricColumns.length}>No testing jobs</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export function TestingComparison({
  jobs,
  modelTaskById = {},
  modelNameById = {},
  taskType
}: {
  jobs: EvaluationJob[];
  modelTaskById?: Record<string, TaskType>;
  modelNameById?: Record<string, string>;
  taskType?: TaskType;
}) {
  if (jobs.length === 0) return <EmptyState label="No completed results for this dataset" icon={<FlaskConical size={28} />} />;
  const metricColumns = comparisonMetricColumns(jobs, modelTaskById, taskType);
  return (
    <div className="table-wrap responsive-card-table">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Type</th>
            <th>Samples</th>
            {metricColumns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td data-label="Model">{displayModelName(job.model_id, modelNameById)}</td>
              <td data-label="Type">{evaluationKindLabel(inferEvaluationKind(job, modelTaskById[job.model_id]))}</td>
              <td data-label="Samples">{metricsOf(job).samples ?? "-"}</td>
              {metricColumns.map((column) => (
                <td data-label={column.label} key={column.key}>{formatMetric(column.get(job))}</td>
              ))}
              <td data-label="Open"><Link className="secondary-button" href={`/testing/${job.id}`}>Details</Link></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function testingTableColumns(jobs: EvaluationJob[], modelTaskById: Record<string, TaskType>, taskType?: TaskType) {
  const taskKind = taskType ? evaluationKindFromTask(taskType) : null;
  if (taskKind === "classification") return classificationMetricColumns().slice(0, 2);
  if (taskKind === "text_classification") return textClassificationMetricColumns().slice(0, 2);
  if (taskKind === "summarization") return summarizationMetricColumns().slice(0, 2);
  if (taskKind === "question_answering") return qaMetricColumns().slice(0, 2);
  if (taskKind === "vision") return visionMetricColumns().slice(0, 2);
  if (taskKind === "llm") return llmMetricColumns().slice(0, 2);
  const kinds = new Set(jobs.map((job) => inferEvaluationKind(job, modelTaskById[job.model_id])));
  if (kinds.size === 1 && kinds.has("classification")) {
    return classificationMetricColumns().slice(0, 2);
  }
  if (kinds.size === 1 && kinds.has("text_classification")) return textClassificationMetricColumns().slice(0, 2);
  if (kinds.size === 1 && kinds.has("summarization")) return summarizationMetricColumns().slice(0, 2);
  if (kinds.size === 1 && kinds.has("question_answering")) return qaMetricColumns().slice(0, 2);
  if (kinds.size === 1 && kinds.has("llm")) return llmMetricColumns().slice(0, 2);
  if (kinds.size === 1 && kinds.has("vision")) {
    return visionMetricColumns().slice(0, 2);
  }
  return [
    classificationMetricColumns()[0],
    visionMetricColumns()[0]
  ];
}

function comparisonMetricColumns(jobs: EvaluationJob[], modelTaskById: Record<string, TaskType>, taskType?: TaskType) {
  const taskKind = taskType ? evaluationKindFromTask(taskType) : null;
  if (taskKind === "classification") return classificationMetricColumns();
  if (taskKind === "text_classification") return textClassificationMetricColumns();
  if (taskKind === "summarization") return summarizationMetricColumns();
  if (taskKind === "question_answering") return qaMetricColumns();
  if (taskKind === "vision") return visionMetricColumns();
  if (taskKind === "llm") return llmMetricColumns();
  const kinds = new Set(jobs.map((job) => inferEvaluationKind(job, modelTaskById[job.model_id])));
  if (kinds.size === 1 && kinds.has("classification")) return classificationMetricColumns();
  if (kinds.size === 1 && kinds.has("text_classification")) return textClassificationMetricColumns();
  if (kinds.size === 1 && kinds.has("summarization")) return summarizationMetricColumns();
  if (kinds.size === 1 && kinds.has("question_answering")) return qaMetricColumns();
  if (kinds.size === 1 && kinds.has("llm")) return llmMetricColumns();
  if (kinds.size === 1 && kinds.has("vision")) return visionMetricColumns();
  return [
    ...classificationMetricColumns().slice(0, 2),
    ...visionMetricColumns().slice(0, 2)
  ];
}

function classificationMetricColumns() {
  return [
    { key: "accuracy", label: "Accuracy", get: (job: EvaluationJob) => numberMetric(metricsOf(job).image?.overall?.accuracy) },
    { key: "macro_f1", label: "Macro F1", get: (job: EvaluationJob) => numberMetric(metricsOf(job).image?.overall?.macro_f1) },
    { key: "weighted_f1", label: "Weighted F1", get: (job: EvaluationJob) => numberMetric(metricsOf(job).image?.overall?.weighted_f1) },
    { key: "mcc", label: "MCC", get: (job: EvaluationJob) => numberMetric(metricsOf(job).image?.overall?.mcc) }
  ];
}

function visionMetricColumns() {
  return [
    { key: "pixel_dice", label: "Pixel Dice", get: (job: EvaluationJob) => numberMetric(metricsOf(job).pixel?.dice) },
    { key: "pixel_iou", label: "Pixel IoU", get: (job: EvaluationJob) => numberMetric(metricsOf(job).pixel?.iou) },
    { key: "object_precision", label: "Object Precision", get: (job: EvaluationJob) => numberMetric(metricsOf(job).object?.precision) },
    { key: "object_recall", label: "Object Recall", get: (job: EvaluationJob) => numberMetric(metricsOf(job).object?.recall) }
  ];
}

function textClassificationMetricColumns() {
  return [
    { key: "text_accuracy", label: "Accuracy", get: (job: EvaluationJob) => numberMetric(metricsOf(job).text_classification?.overall?.accuracy) },
    { key: "text_macro_f1", label: "Macro F1", get: (job: EvaluationJob) => numberMetric(metricsOf(job).text_classification?.overall?.macro_f1) },
    { key: "text_weighted_f1", label: "Weighted F1", get: (job: EvaluationJob) => numberMetric(metricsOf(job).text_classification?.overall?.weighted_f1) }
  ];
}

function summarizationMetricColumns() {
  return [
    { key: "rouge1", label: "ROUGE-1", get: (job: EvaluationJob) => numberMetric(metricsOf(job).summarization?.rouge1) },
    { key: "rouge2", label: "ROUGE-2", get: (job: EvaluationJob) => numberMetric(metricsOf(job).summarization?.rouge2) },
    { key: "rougeL", label: "ROUGE-L", get: (job: EvaluationJob) => numberMetric(metricsOf(job).summarization?.rougeL) }
  ];
}

function qaMetricColumns() {
  return [
    { key: "exact_match", label: "Exact match", get: (job: EvaluationJob) => numberMetric(metricsOf(job).question_answering?.exact_match) },
    { key: "qa_f1", label: "F1", get: (job: EvaluationJob) => numberMetric(metricsOf(job).question_answering?.f1) }
  ];
}

function llmMetricColumns() {
  return [
    { key: "perplexity", label: "Perplexity", get: (job: EvaluationJob) => numberMetric(metricsOf(job).llm?.perplexity) },
    { key: "token_accuracy", label: "Token accuracy", get: (job: EvaluationJob) => numberMetric(metricsOf(job).llm?.token_accuracy) },
    { key: "loss", label: "Loss", get: (job: EvaluationJob) => numberMetric(metricsOf(job).llm?.loss) }
  ];
}

function detailMetricEntries(job: EvaluationJob, kind: EvaluationDisplayKind): MetricEntry[] {
  const columns = kind === "classification"
    ? classificationMetricColumns()
    : kind === "text_classification"
      ? textClassificationMetricColumns()
      : kind === "summarization"
        ? summarizationMetricColumns()
        : kind === "question_answering"
          ? qaMetricColumns()
          : kind === "llm"
            ? llmMetricColumns()
            : visionMetricColumns();
  const entries = columns
    .map((column) => ({ key: column.key, label: column.label, value: column.get(job) }))
    .filter((entry): entry is MetricEntry => typeof entry.value === "number");
  if (kind === "classification") {
    const macroAuc = numberMetric(metricsOf(job).classification?.macro_auc);
    const balancedAccuracy = numberMetric(metricsOf(job).image?.overall?.balanced_accuracy);
    return [
      ...(typeof balancedAccuracy === "number" ? [{ key: "balanced_accuracy", label: "Balanced Acc.", value: balancedAccuracy }] : []),
      ...entries,
      ...(typeof macroAuc === "number" ? [{ key: "macro_auc", label: "Macro AUC", value: macroAuc }] : [])
    ];
  }
  return entries;
}

function inferEvaluationKind(job: EvaluationJob, modelTask?: TaskType): EvaluationDisplayKind {
  if (modelTask === "classification") return "classification";
  if (modelTask === "object_detection" || modelTask === "segmentation") return "vision";
  if (modelTask === "text_classification" || modelTask === "summarization" || modelTask === "question_answering") return modelTask;
  if (modelTask === "llm_finetune") return "llm";
  if (metricsOf(job).llm) return "llm";
  if (metricsOf(job).pixel || metricsOf(job).object) return "vision";
  if (metricsOf(job).text_classification) return "text_classification";
  if (metricsOf(job).summarization) return "summarization";
  if (metricsOf(job).question_answering) return "question_answering";
  return "classification";
}

function evaluationKindFromTask(taskType: TaskType): EvaluationDisplayKind {
  if (taskType === "classification") return "classification";
  if (taskType === "text_classification") return "text_classification";
  if (taskType === "summarization") return "summarization";
  if (taskType === "question_answering") return "question_answering";
  if (taskType === "llm_finetune") return "llm";
  return "vision";
}

function evaluationKindLabel(kind: EvaluationDisplayKind): string {
  if (kind === "classification") return "Classification";
  if (kind === "vision") return "Detection / segmentation";
  if (kind === "text_classification") return "Text classification";
  if (kind === "summarization") return "Summarization";
  if (kind === "llm") return "LLM (perplexity)";
  return "Question answering";
}

function numberMetric(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

// `job.metrics` is a genuinely dynamic dict (shape varies per task type), so callers
// read nested/arbitrary properties off it via a local cast rather than a schema change.
function metricsOf(job: EvaluationJob): Record<string, any> {
  return (job.metrics ?? {}) as Record<string, any>;
}

export function MetricsDetails({ job, rows, modelTask }: { job: EvaluationJob; rows: EvaluationPerImageRow[]; modelTask?: TaskType }) {
  const metrics = metricsOf(job);
  if (!metrics.samples) return <EmptyState label="No metrics yet" icon={<FlaskConical size={28} />} />;
  const kind = inferEvaluationKind(job, modelTask);
  const labels = metrics.labels ?? metrics.image?.labels ?? metrics.text_classification?.labels ?? [];
  const scoreEntries = detailMetricEntries(job, kind);
  return (
    <div className="space-y-5">
      <div className="metric-grid">
        <Metric label="Samples" value={metrics.samples} />
        {scoreEntries.slice(0, 5).map((entry) => (
          <Metric key={entry.key} label={entry.label} value={formatMetric(entry.value)} />
        ))}
      </div>
      {kind === "vision" && <ObjectSummaryPanel objectMetrics={metrics.object ?? {}} />}
      {(kind === "classification" || kind === "text_classification" || kind === "vision") && (
        <div className="grid gap-4 xl:grid-cols-2">
          <ConfusionMatrix title={kind === "text_classification" ? "Text Confusion" : "Image Confusion"} labels={metrics.image?.labels ?? metrics.text_classification?.labels ?? labels} matrix={metrics.image?.confusion_matrix ?? metrics.text_classification?.confusion_matrix ?? []} />
          {kind === "vision" && (
            <ConfusionMatrix title="Object Confusion" labels={[...labels, "background"]} matrix={metrics.object?.confusion_matrix ?? []} />
          )}
        </div>
      )}
      <PerClassReport report={metrics.image?.report ?? metrics.text_classification?.report ?? {}} />
      <PerImageTable rows={rows} labels={[...labels, "Normal"]} kind={kind} />
    </div>
  );
}

function ObjectSummaryPanel({ objectMetrics }: { objectMetrics: Record<string, any> }) {
  const entries = [
    ["Matched", objectMetrics.matched],
    ["Ground truth", objectMetrics.ground_truth_objects],
    ["Predicted", objectMetrics.predicted_objects],
    ["IoU threshold", objectMetrics.iou_threshold],
  ].filter((entry): entry is [string, number] => typeof entry[1] === "number");
  if (entries.length === 0) return null;
  return (
    <div className="metric-grid">
      {entries.map(([label, value]) => (
        <Metric key={label} label={label} value={formatMetric(value)} />
      ))}
    </div>
  );
}

function ConfusionMatrix({ title, labels, matrix }: { title: string; labels: string[]; matrix: number[][] }) {
  if (!matrix.length) return <EmptyState label={title} icon={<FlaskConical size={28} />} />;
  return (
    <div>
      <h3 className="section-title">{title}</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th />
              {labels.map((label) => <th key={label}>{label}</th>)}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, index) => (
              <tr key={index}>
                <th>{labels[index] ?? index}</th>
                {row.map((value, valueIndex) => <td key={valueIndex}>{value}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PerClassReport({ report }: { report: Record<string, any> }) {
  const rows = Object.entries(report).filter(([, value]) => typeof value === "object");
  if (rows.length === 0) return null;
  return (
    <div>
      <h3 className="section-title">Class Report</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, value]) => (
              <tr key={label}>
                <td>{label}</td>
                <td>{formatMetric(value.precision)}</td>
                <td>{formatMetric(value.recall)}</td>
                <td>{formatMetric(value["f1-score"])}</td>
                <td>{value.support ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PerImageTable({
  rows,
  labels,
  kind
}: {
  rows: EvaluationPerImageRow[];
  labels: string[];
  kind: EvaluationDisplayKind;
}) {
  if (rows.length === 0) return null;
  const classification = kind === "classification";
  const textClassification = kind === "text_classification";
  const generativeText = kind === "summarization" || kind === "question_answering";
  const llm = kind === "llm";
  if (llm) {
    return (
      <div>
        <h3 className="section-title">Per Record</h3>
        <div className="table-wrap max-h-[420px] overflow-y-auto">
          <table>
            <thead>
              <tr>
                <th>Prompt</th>
                <th>Reference</th>
                <th>Perplexity</th>
                <th>Loss</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.image}>
                  <td>{row.text_preview || row.image}</td>
                  <td>{row.reference_text ?? "-"}</td>
                  <td>{formatMetric(row.scores?.perplexity)}</td>
                  <td>{formatMetric(row.scores?.loss)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
  return (
    <div>
      <h3 className="section-title">{kind === "vision" || kind === "classification" ? "Per Image" : "Per Text"}</h3>
      <div className="table-wrap max-h-[420px] overflow-y-auto">
        <table>
          <thead>
            {classification || textClassification ? (
              <tr>
                <th>{textClassification ? "Text" : "Image"}</th>
                <th>Ground truth class</th>
                <th>Predicted class</th>
                <th>Accuracy score</th>
              </tr>
            ) : generativeText ? (
              <tr>
                <th>Text</th>
                <th>Reference</th>
                <th>Prediction</th>
                <th>Score</th>
              </tr>
            ) : (
              <tr>
                <th>Image</th>
                <th>GT</th>
                <th>Pred</th>
                <th>Dice</th>
                <th>Matched</th>
                <th>FP</th>
                <th>FN</th>
              </tr>
            )}
          </thead>
          <tbody>
            {rows.map((row) => {
              const accuracyScore = row.ground_truth === row.prediction ? 1 : 0;
              return classification || textClassification ? (
                <tr key={row.image}>
                  <td>{textClassification ? (row.text_preview || row.image) : row.image}</td>
                  <td>{labels[row.ground_truth] ?? row.ground_truth}</td>
                  <td>{labels[row.prediction] ?? row.prediction}</td>
                  <td>{formatMetric(accuracyScore)}</td>
                </tr>
              ) : generativeText ? (
                <tr key={row.image}>
                  <td>{row.text_preview || row.image}</td>
                  <td>{row.reference_text ?? "-"}</td>
                  <td>{row.prediction_text ?? "-"}</td>
                  <td>{formatMetric(kind === "summarization" ? row.scores?.rougeL : row.scores?.f1)}</td>
                </tr>
              ) : (
                <tr key={row.image}>
                  <td>{row.image}</td>
                  <td>{labels[row.ground_truth] ?? row.ground_truth}</td>
                  <td>{labels[row.prediction] ?? row.prediction}</td>
                  <td>{formatMetric(row.pixel?.dice)}</td>
                  <td>{row.object?.matched ?? "-"}</td>
                  <td>{row.object?.false_positives ?? "-"}</td>
                  <td>{row.object?.false_negatives ?? "-"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
