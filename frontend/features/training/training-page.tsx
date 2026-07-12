"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, Play, Upload } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { TrainingJobTable } from "@/features/training/training-components";
import { listPollInterval } from "@/features/platform/utils";
import { CardGridSkeleton, Field, HistoryHeader, MutationError, PageHeader, PanelTitle, TableSkeleton, useConfirmationDialog } from "@/features/platform/ui";
import type { TaskType, TrainingJob } from "@/types/api";

export function TrainingPage() {
  const { projectId } = useProject();
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [modelOptionId, setModelOptionId] = useState("");
  const [modelName, setModelName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [epochs, setEpochs] = useState(50);
  const [imageSize, setImageSize] = useState(512);
  const [batchSize, setBatchSize] = useState(16);
  const [optimizer, setOptimizer] = useState("AdamW");
  const [learningRate, setLearningRate] = useState(0.002);
  const [device, setDevice] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const datasetsQuery = useQuery({ queryKey: ["training-datasets", projectId], queryFn: () => api.datasetCatalog(projectId) });
  const optionsQuery = useQuery({
    queryKey: ["training-options", taskType],
    queryFn: () => api.trainingOptions(taskType)
  });
  const jobsQuery = useQuery({
    queryKey: ["training-jobs", projectId],
    queryFn: () => api.trainingJobs(projectId),
    refetchInterval: (query) => listPollInterval(query.state.data as TrainingJob[] | undefined)
  });
  const createMutation = useMutation({
    mutationFn: api.createTrainingJob,
    onSuccess: async () => {
      setModelName("");
      await jobsQuery.refetch();
    }
  });
  const prepareMutation = useMutation({ mutationFn: () => api.prepareModelAsset(modelOptionId, true) });
  const deleteMutation = useMutation({
    mutationFn: ({ ids, clearAll }: { ids: string[]; clearAll?: boolean }) =>
      api.deleteTrainingJobs(ids, projectId, clearAll),
    onSuccess: async () => {
      setSelectedIds([]);
      await jobsQuery.refetch();
    }
  });
  const options = useMemo(() => optionsQuery.data ?? [], [optionsQuery.data]);
  const option = options.find((item) => item.id === modelOptionId);
  const datasets = useMemo(
    () => (datasetsQuery.data ?? []).filter((dataset) => dataset.task_type === taskType),
    [datasetsQuery.data, taskType]
  );

  useEffect(() => {
    if (!options.some((item) => item.id === modelOptionId)) {
      const firstRunnable = options.find((item) => item.runnable) ?? options[0];
      if (firstRunnable) setModelOptionId(firstRunnable.id);
    }
  }, [modelOptionId, options]);

  useEffect(() => {
    if (!option) return;
    setImageSize((value) => Number(option.defaults.image_size ?? value));
    setOptimizer((value) => String(option.defaults.optimizer ?? value));
    setLearningRate((value) => Number(option.defaults.learning_rate ?? value));
  }, [option]);

  useEffect(() => {
    if (datasets.length === 0) {
      setDatasetId("");
      return;
    }
    if (!datasets.some((dataset) => dataset.id === datasetId)) setDatasetId(datasets[0].id);
  }, [datasetId, datasets]);

  async function runTraining() {
    const selectedOption = option ?? options[0];
    if (!selectedOption || !datasetId) return;
    await createMutation.mutateAsync({
      project_id: projectId,
      task_type: taskType,
      model_family: selectedOption.family,
      model_option_id: selectedOption.id,
      model_name: modelName.trim() || null,
      epochs,
      image_size: imageSize,
      batch_size: batchSize,
      dataset_id: datasetId,
      optimizer,
      learning_rate: learningRate,
      device,
      cache: "disk",
      workers: 0,
      patience: 50
    });
  }

  function confirmDeleteTrainingRows() {
    if (selectedIds.length === 0) return;
    confirm({
      title: "Delete training jobs?",
      message: `This will delete ${selectedIds.length} selected terminal training job${selectedIds.length === 1 ? "" : "s"} and owned artifacts.`,
      confirmLabel: "Delete jobs",
      onConfirm: () => deleteMutation.mutate({ ids: selectedIds })
    });
  }

  function confirmClearTrainingJobs() {
    confirm({
      title: "Clear all training jobs?",
      message: "This will delete all terminal training jobs for the active project and owned artifacts.",
      confirmLabel: "Clear all",
      onConfirm: () => deleteMutation.mutate({ ids: [], clearAll: true })
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Training" subtitle="Configure model runs" icon={<Activity size={20} />} />
      <div className="workspace-grid workspace-grid-training">
        <section className="panel">
          <PanelTitle icon={<Activity size={18} />} title="New Run" />
          {(optionsQuery.isLoading || datasetsQuery.isLoading) && <CardGridSkeleton count={1} />}
          <Field label="Model name">
            <input value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="Optional display name" />
          </Field>
          <Field label="Task">
            <select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}>
              <option value="classification">Classification</option>
              <option value="object_detection">Object detection</option>
              <option value="segmentation">Segmentation</option>
            </select>
          </Field>
          <Field label="Base model">
            <select value={modelOptionId} onChange={(event) => setModelOptionId(event.target.value)}>
              {options.map((item) => (
                <option key={item.id} value={item.id} disabled={!item.runnable}>
                  {item.name}
                  {item.runnable ? "" : " (gated)"}
                </option>
              ))}
            </select>
          </Field>
          {option && <p className="text-sm text-slate-500">{option.description}</p>}
          <Field label="Dataset">
            <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
              {datasets.map((dataset) => (
                <option value={dataset.id} key={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </Field>
          <div className="form-grid form-grid-three">
            <Field label="Epochs">
              <input type="number" min={1} max={1000} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} />
            </Field>
            <Field label="Size">
              <input type="number" min={128} max={2048} value={imageSize} onChange={(event) => setImageSize(Number(event.target.value))} />
            </Field>
            <Field label="Batch">
              <input type="number" min={-1} max={256} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
            </Field>
          </div>
          <div className="form-grid form-grid-two">
            <Field label="Optimizer">
              <select value={optimizer} onChange={(event) => setOptimizer(event.target.value)}>
                <option value="AdamW">AdamW</option>
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
              </select>
            </Field>
            <Field label="LR">
              <input type="number" step={0.0001} value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} />
            </Field>
          </div>
          <Field label="Device">
            <input value={device} onChange={(event) => setDevice(event.target.value)} />
          </Field>
          <div className="action-row action-row-split">
            <button className="secondary-button" onClick={() => prepareMutation.mutate()} disabled={!option?.needs_download || prepareMutation.isPending}>
              <Upload size={16} /> Prepare
            </button>
            <button className="primary-button flex-1" onClick={runTraining} disabled={createMutation.isPending || !option?.runnable || !datasetId}>
              <Play size={17} /> Start training
            </button>
          </div>
          {prepareMutation.data && <p className="text-sm text-slate-500">{prepareMutation.data.status}: {prepareMutation.data.message ?? prepareMutation.data.path}</p>}
          <MutationError mutations={[createMutation, prepareMutation]} />
        </section>
        <section className="panel">
          <HistoryHeader
            title="Training Jobs"
            selectedCount={selectedIds.length}
            onRefresh={() => jobsQuery.refetch()}
            onDelete={confirmDeleteTrainingRows}
            onClear={confirmClearTrainingJobs}
          />
          <TrainingJobTable jobs={jobsQuery.data ?? []} selectedIds={selectedIds} setSelectedIds={setSelectedIds} />
          {jobsQuery.isLoading && <TableSkeleton rows={5} />}
          <MutationError mutations={[deleteMutation]} />
        </section>
      </div>
      {confirmationDialog}
    </div>
  );
}

