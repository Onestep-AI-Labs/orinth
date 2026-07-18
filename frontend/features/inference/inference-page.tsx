"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useState } from "react";
import { Play, ScanEye, Upload } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, mediaUrl } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { NLP_TASK_TYPES, VISION_TASK_TYPES, activePollInterval, displayModelName, formatDatasetTask, formatMetric, isNlpTask, labelColor } from "@/features/platform/utils";
import {
  Button,
  ButtonLink,
  CardGridSkeleton,
  EmptyState,
  Field,
  HistoryHeader,
  MutationError,
  PageHeader,
  PanelTitle,
  ProgressPanel,
  SliderField,
  TableSkeleton,
  toggleId,
  useConfirmationDialog
} from "@/features/platform/ui";
import type { InferenceJob, InferenceResult, ModelInfo, TaskType } from "@/types/api";

function InferencePageInner({ models, modelsLoading }: { models: ModelInfo[]; modelsLoading?: boolean }) {
  const { projectId, project } = useProject();
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [selectedModel, setSelectedModel] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [textContent, setTextContent] = useState("");
  const [question, setQuestion] = useState("");
  const [confidence, setConfidence] = useState(0.65);
  const [iou, setIou] = useState(0.7);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [activeJobId, setActiveJobId] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const taskModels = useMemo(
    () => models.filter((model) => model.task_type === taskType),
    [models, taskType]
  );
  const selectedModelInfo = useMemo(
    () => taskModels.find((model) => model.id === selectedModel) ?? null,
    [taskModels, selectedModel]
  );
  const modelNameById = useMemo(
    () => Object.fromEntries(models.map((model) => [model.id, model.name])) as Record<string, string>,
    [models]
  );
  const taskOptions = useMemo(() => {
    const projectTasks = (project?.task_types ?? []).filter((task): task is TaskType => [...VISION_TASK_TYPES, ...NLP_TASK_TYPES].includes(task));
    const modelTasks = [...new Set(models.map((model) => model.task_type))] as TaskType[];
    return projectTasks.length ? projectTasks : modelTasks.length ? modelTasks : VISION_TASK_TYPES;
  }, [models, project?.task_types]);
  const nlp = isNlpTask(taskType);
  const showDetectionParams = selectedModelInfo ? selectedModelInfo.task_type !== "classification" : false;
  const historyQuery = useQuery({
    queryKey: ["inference-history", projectId],
    queryFn: () => api.inferenceHistory(projectId)
  });
  const jobQuery = useQuery({
    queryKey: ["inference-job", activeJobId],
    queryFn: () => api.inferenceJob(activeJobId),
    enabled: Boolean(activeJobId),
    refetchInterval: (query) => activePollInterval(query.state.data as InferenceJob | undefined)
  });
  const inferenceMutation = useMutation({
    mutationFn: api.createInferenceJob,
    onSuccess: (job) => setActiveJobId(job.id)
  });
  const deleteMutation = useMutation({
    mutationFn: ({ ids, clearAll }: { ids: string[]; clearAll?: boolean }) =>
      api.deleteInference(ids, projectId, clearAll),
    onSuccess: async () => {
      setSelectedIds([]);
      await historyQuery.refetch();
    }
  });

  useEffect(() => {
    if (models.length > 0 && !models.some((model) => model.task_type === taskType)) {
      setTaskType(models[0].task_type);
    }
    if (!taskOptions.includes(taskType) && taskOptions.length > 0) setTaskType(taskOptions[0]);
  }, [models, taskOptions, taskType]);

  useEffect(() => {
    if (taskModels.length === 0) {
      if (selectedModel) setSelectedModel("");
      return;
    }
    if (!taskModels.some((model) => model.id === selectedModel)) setSelectedModel(taskModels[0].id);
  }, [taskModels, selectedModel]);

  useEffect(() => {
    if (jobQuery.data?.result) {
      setResult(jobQuery.data.result);
      historyQuery.refetch();
    }
  }, [jobQuery.data?.result, historyQuery]);

  async function runInference() {
    if ((!file && !nlp) || (!textContent.trim() && nlp) || !selectedModel) return;
    const form = new FormData();
    if (file && !nlp) form.append("file", file);
    if (nlp) {
      form.append("text_content", textContent);
      if (question.trim()) form.append("question", question.trim());
    }
    form.append("project_id", projectId);
    form.append("model_id", selectedModel);
    if (showDetectionParams && !nlp) {
      form.append("confidence_threshold", String(confidence));
      form.append("iou_threshold", String(iou));
    }
    await inferenceMutation.mutateAsync(form);
  }

  function confirmDeleteInferenceRows() {
    if (selectedIds.length === 0) return;
    confirm({
      title: "Delete inference history?",
      message: `This will delete ${selectedIds.length} selected inference result${selectedIds.length === 1 ? "" : "s"} and owned output artifacts.`,
      confirmLabel: "Delete results",
      onConfirm: () => deleteMutation.mutate({ ids: selectedIds })
    });
  }

  function confirmClearInferenceHistory() {
    confirm({
      title: "Clear all inference history?",
      message: "This will delete all inference history for the active project and owned output artifacts.",
      confirmLabel: "Clear all",
      onConfirm: () => deleteMutation.mutate({ ids: [], clearAll: true })
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Inference" subtitle="Run models on project samples" icon={<ScanEye size={20} />} />
      <div className="workspace-grid workspace-grid-inference">
        <section className="panel">
          <PanelTitle icon={<Upload size={18} />} title="Run" />
          {modelsLoading && <CardGridSkeleton count={1} />}
          <Field label="Task">
            <select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}>
              {taskOptions.map((task) => (
                <option value={task} key={task}>{formatDatasetTask(task)}</option>
              ))}
            </select>
          </Field>
          <Field label="Model">
            <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              {taskModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} - {formatDatasetTask(model.task_type)}
                </option>
              ))}
            </select>
            {taskModels.length === 0 && (
              <span className="field-hint">
                No available models yet — <ButtonLink variant="ghost" size="sm" href="/training">train one</ButtonLink>
              </span>
            )}
          </Field>
          {selectedModelInfo && (
            <div className="inference-model-meta">
              <span><strong>Task</strong>{formatDatasetTask(selectedModelInfo.task_type)}</span>
              <span><strong>Source</strong>{selectedModelInfo.source}</span>
              <span><strong>Labels</strong>{selectedModelInfo.labels.length}</span>
            </div>
          )}
          {nlp ? (
            <>
              <Field label={taskType === "question_answering" ? "Context" : "Text"}>
                <textarea value={textContent} onChange={(event) => setTextContent(event.target.value)} rows={7} />
              </Field>
              {taskType === "question_answering" && (
                <Field label="Question">
                  <input value={question} onChange={(event) => setQuestion(event.target.value)} />
                </Field>
              )}
            </>
          ) : (
            <Field label="Image">
              <input type="file" accept="image/png,image/jpeg,image/jpg,image/webp" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            </Field>
          )}
          {showDetectionParams && !nlp && (
            <>
              <SliderField label="Confidence" value={confidence} min={0} max={1} step={0.01} onChange={setConfidence} />
              <SliderField label="IoU" value={iou} min={0} max={1} step={0.01} onChange={setIou} />
            </>
          )}
          <Button className="mt-2 w-full" disabled={(!file && !nlp) || (nlp && !textContent.trim()) || !selectedModel || inferenceMutation.isPending} onClick={runInference}>
            <Play size={17} /> Run inference
          </Button>
          {inferenceMutation.error && <p className="error-text">{inferenceMutation.error.message}</p>}
          {jobQuery.data && <ProgressPanel progress={jobQuery.data.progress} status={jobQuery.data.status} error={jobQuery.data.error} />}
        </section>
        <section className="panel min-h-[520px]">
          <PanelTitle icon={<ScanEye size={18} />} title="Result" />
          {result ? (
            <InferenceResultView result={result} modelNameById={modelNameById} />
          ) : (
            <EmptyState
              label="No result selected"
              icon={<ScanEye size={28} />}
              description="Run an inference or pick a row from the history below."
            />
          )}
        </section>
        <section className="panel workspace-grid-full">
          <HistoryHeader
            title="Inference History"
            selectedCount={selectedIds.length}
            onRefresh={() => historyQuery.refetch()}
            onDelete={confirmDeleteInferenceRows}
            onClear={confirmClearInferenceHistory}
          />
          <InferenceHistory
            rows={historyQuery.data ?? []}
            selectedIds={selectedIds}
            setSelectedIds={setSelectedIds}
            onSelect={setResult}
            modelNameById={modelNameById}
          />
          {historyQuery.isLoading && <TableSkeleton rows={4} />}
          <MutationError mutations={[deleteMutation]} />
        </section>
      </div>
      {confirmationDialog}
    </div>
  );
}

export function InferencePage() {
  const { projectId } = useProject();
  const modelsQuery = useQuery({
    queryKey: ["models", "available", projectId],
    queryFn: () => api.models(true, projectId)
  });
  return <InferencePageInner models={modelsQuery.data ?? []} modelsLoading={modelsQuery.isLoading} />;
}

function InferenceResultView({ result, modelNameById = {} }: { result: InferenceResult; modelNameById?: Record<string, string> }) {
  const overlay = mediaUrl(result.overlay_url);
  const modelName = displayModelName(result.model_id, modelNameById);
  if (result.input_type === "text") {
    return (
      <div className="result-grid result-grid-text">
        <div className="text-item-preview result-text-input">{result.text_content}</div>
        <div className="inference-result-side">
          <div className="inference-result-summary">
            <span><strong>Model</strong>{modelName}</span>
            <span><strong>Result</strong>{result.image_level_label}</span>
            <span><strong>Time</strong>{result.duration_ms ? `${result.duration_ms} ms` : "-"}</span>
          </div>
          <NlpResultView result={result} />
        </div>
      </div>
    );
  }
  return (
    <div className="result-grid">
      <div className="image-frame">{overlay ? <img src={overlay} alt="Prediction overlay" /> : null}</div>
      <div className="inference-result-side">
        <div className="inference-result-summary">
          <span><strong>Model</strong>{modelName}</span>
          <span><strong>Image label</strong>{result.image_level_label}</span>
          <span><strong>Detections</strong>{result.detections.length}</span>
          <span><strong>Time</strong>{result.duration_ms ? `${result.duration_ms} ms` : "-"}</span>
        </div>
        {Object.keys(result.class_scores ?? {}).length > 0 ? (
          <ClassScoreTable scores={result.class_scores} />
        ) : (
          <DetectionTable result={result} />
        )}
      </div>
    </div>
  );
}

function NlpResultView({ result }: { result: InferenceResult }) {
  const payload = (result.nlp_result ?? {}) as Record<string, any>;
  if (payload.scores) return <ClassScoreTable scores={payload.scores} />;
  if (payload.summary) {
    return (
      <div className="nlp-result-card">
        <strong>Summary</strong>
        <p>{payload.summary}</p>
      </div>
    );
  }
  if (payload.answer) {
    return (
      <div className="nlp-result-card">
        <strong>Answer</strong>
        <p>{payload.answer}</p>
        {typeof payload.score === "number" && <span>Score {formatMetric(payload.score)}</span>}
      </div>
    );
  }
  return <pre className="log-box">{JSON.stringify(payload, null, 2)}</pre>;
}

function ClassScoreTable({ scores }: { scores: Record<string, number> }) {
  const maxScore = Math.max(0.0001, ...Object.values(scores));
  return (
    <div className="table-wrap mt-4">
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Score</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {Object.entries(scores).map(([label, score], index) => (
            <tr key={label}>
              <td><span className="class-dot" style={{ backgroundColor: labelColor(index) }} />{label}</td>
              <td>{formatMetric(score)}</td>
              <td>
                <div className="eda-bar-track">
                  <div className="eda-bar-fill" style={{ width: `${(score / maxScore) * 100}%`, backgroundColor: labelColor(index) }} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DetectionTable({ result }: { result: InferenceResult }) {
  return (
    <div className="table-wrap mt-4">
      <table>
        <thead>
          <tr>
            <th>Label</th>
            <th>Conf.</th>
            <th>Area</th>
            <th>Box</th>
          </tr>
        </thead>
        <tbody>
          {result.detections.map((detection, index) => (
            <tr key={`${detection.class_name}-${index}`}>
              <td>{detection.class_name}</td>
              <td>{detection.confidence.toFixed(2)}</td>
              <td>{Math.round(detection.mask_area)}</td>
              <td>
                {Math.round(detection.bbox.width)} x {Math.round(detection.bbox.height)}
              </td>
            </tr>
          ))}
          {result.detections.length === 0 && (
            <tr>
              <td colSpan={4}>No detections</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function InferenceHistory({
  rows,
  selectedIds,
  setSelectedIds,
  onSelect,
  modelNameById = {}
}: {
  rows: InferenceResult[];
  selectedIds: string[];
  setSelectedIds: (ids: string[]) => void;
  onSelect: (result: InferenceResult) => void;
  modelNameById?: Record<string, string>;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th />
            <th>Time</th>
            <th>Model</th>
              <th>Label</th>
              <th>Detections</th>
            <th>ms</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="click-row">
              <td data-label="Select">
                <input
                  type="checkbox"
                  checked={selectedIds.includes(row.id)}
                  onChange={(event) => toggleId(row.id, event.target.checked, selectedIds, setSelectedIds)}
                />
              </td>
              <td onClick={() => onSelect(row)}>{new Date(row.created_at).toLocaleString()}</td>
              <td onClick={() => onSelect(row)}>{displayModelName(row.model_id, modelNameById)}</td>
              <td onClick={() => onSelect(row)}>{row.image_level_label}</td>
              <td onClick={() => onSelect(row)}>{row.input_type === "text" ? "text" : row.detections.length}</td>
              <td onClick={() => onSelect(row)}>{row.duration_ms ?? "-"}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={6}>No inference history</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
