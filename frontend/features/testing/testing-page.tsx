"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, FlaskConical, Play } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { TestingComparison, TestingJobTable } from "@/features/testing/testing-components";
import { allowedTaskTypesForProject, areTasksCompatible, listPollInterval } from "@/features/platform/utils";
import {
  ButtonLink,
  CardGridSkeleton,
  EmptyState,
  Field,
  HistoryHeader,
  MultiSelect,
  MutationError,
  PageHeader,
  PanelTitle,
  TableSkeleton,
  TaskSelect,
  useConfirmationDialog
} from "@/features/platform/ui";
import type { EvaluationJob, TaskType } from "@/types/api";

export function TestingPage() {
  const { projectId, project } = useProject();
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [modelIds, setModelIds] = useState<string[]>([]);
  const [datasetKey, setDatasetKey] = useState("");
  const [limit, setLimit] = useState<number | "">("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const modelsQuery = useQuery({
    queryKey: ["models", "available", projectId],
    queryFn: () => api.models(true, projectId)
  });
  const datasetsQuery = useQuery({ queryKey: ["testing-datasets", projectId, taskType], queryFn: () => api.datasets(projectId, taskType) });
  const jobsQuery = useQuery({
    queryKey: ["testing-jobs", projectId],
    queryFn: () => api.testingJobs(projectId),
    refetchInterval: (query) => listPollInterval(query.state.data as EvaluationJob[] | undefined)
  });
  const mutation = useMutation({
    mutationFn: api.createTestingJobsBatch,
    onSuccess: async () => {
      await jobsQuery.refetch();
    }
  });
  const deleteMutation = useMutation({
    mutationFn: ({ ids, clearAll }: { ids: string[]; clearAll?: boolean }) =>
      api.deleteTestingJobs(ids, projectId, clearAll),
    onSuccess: async () => {
      setSelectedIds([]);
      await jobsQuery.refetch();
    }
  });
  const rawModels = useMemo(() => modelsQuery.data ?? [], [modelsQuery.data]);
  const rawDatasets = useMemo(() => datasetsQuery.data ?? [], [datasetsQuery.data]);
  const taskOptions = useMemo(() => allowedTaskTypesForProject(project), [project]);
  const datasets = useMemo(
    () => rawDatasets.filter((dataset) => dataset.task_type === taskType),
    [rawDatasets, taskType]
  );
  const models = useMemo(
    () =>
      rawModels.filter(
        (model) =>
          areTasksCompatible(model.task_type, taskType) &&
          // GGUF models are tested interactively via serving/chat, not batch
          // perplexity eval, so they are not offered on the testing page.
          model.family !== "llm_gguf"
      ),
    [rawModels, taskType]
  );
  const modelTaskById = useMemo(
    () => Object.fromEntries(rawModels.map((model) => [model.id, model.task_type])) as Record<string, TaskType>,
    [rawModels]
  );
  const modelNameById = useMemo(
    () => Object.fromEntries(rawModels.map((model) => [model.id, model.name])) as Record<string, string>,
    [rawModels]
  );
  const datasetTaskByKey = useMemo(
    () => Object.fromEntries(rawDatasets.map((dataset) => [dataset.key, dataset.task_type])) as Record<string, TaskType>,
    [rawDatasets]
  );
  const jobsForTask = useMemo(
    () => (jobsQuery.data ?? []).filter((job) => datasetTaskByKey[job.dataset_key] === taskType),
    [datasetTaskByKey, jobsQuery.data, taskType]
  );
  const comparisonJobs = useMemo(
    () =>
      jobsForTask
        .filter((job) => job.dataset_key === datasetKey && job.status === "completed")
        .slice(0, 8),
    [datasetKey, jobsForTask]
  );

  useEffect(() => {
    if (!taskOptions.includes(taskType)) {
      setTaskType(taskOptions[0] ?? "classification");
    }
  }, [taskOptions, taskType]);
  useEffect(() => {
    setSelectedIds([]);
  }, [taskType]);
  useEffect(() => {
    // Drop models that vanished, but never preselect one: an auto-checked model
    // is easy to miss and gets tested by accident.
    setModelIds((current) => {
      const availableIds = new Set(models.map((model) => model.id));
      return current.filter((id) => availableIds.has(id));
    });
  }, [models]);
  useEffect(() => {
    const availableDataset = datasets.find((dataset) => dataset.available) ?? datasets[0];
    if (!availableDataset) {
      if (datasetKey) setDatasetKey("");
      return;
    }
    if (!datasets.some((dataset) => dataset.key === datasetKey)) {
      setDatasetKey(availableDataset.key);
    }
  }, [datasetKey, datasets]);

  async function runTesting() {
    if (modelIds.length === 0 || !datasetKey) return;
    await mutation.mutateAsync({
      project_id: projectId,
      model_ids: modelIds,
      dataset_key: datasetKey,
      limit: limit === "" ? null : limit
    });
  }

  function confirmDeleteTestingRows() {
    if (selectedIds.length === 0) return;
    confirm({
      title: "Delete testing jobs?",
      message: `This will delete ${selectedIds.length} selected terminal testing job${selectedIds.length === 1 ? "" : "s"} and owned artifacts.`,
      confirmLabel: "Delete jobs",
      onConfirm: () => deleteMutation.mutate({ ids: selectedIds })
    });
  }

  function confirmClearTestingJobs() {
    confirm({
      title: "Clear all testing jobs?",
      message: "This will delete all terminal testing jobs for the active project and owned artifacts.",
      confirmLabel: "Clear all",
      onConfirm: () => deleteMutation.mutate({ ids: [], clearAll: true })
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Testing" subtitle="Evaluate model runs" icon={<FlaskConical size={20} />} />
      <div className="workspace-grid workspace-grid-testing">
        <section className="panel">
          <PanelTitle icon={<FlaskConical size={18} />} title="New Test" dataTour="testing-new" />
          {(modelsQuery.isLoading || datasetsQuery.isLoading) && <CardGridSkeleton count={1} />}
          <Field label="Task">
            <TaskSelect value={taskType} onChange={setTaskType} options={taskOptions} />
          </Field>
          <Field label="Models">
            {models.length === 0 ? (
              <EmptyState
                label="No trained models available"
                icon={<FlaskConical size={28} />}
                description="Finish a training run to evaluate it here."
                action={
                  <ButtonLink variant="secondary" size="sm" href="/training">
                    Go to training
                  </ButtonLink>
                }
              />
            ) : (
              // Several models can be tested against one dataset to produce a
              // comparison run, so this stays multi-select — just closed by
              // default instead of an always-open list.
              <MultiSelect
                options={models.map((model) => ({ value: model.id, label: model.name }))}
                values={modelIds}
                onChange={setModelIds}
                placeholder="Choose one or more models…"
              />
            )}
          </Field>
          <Field label="Dataset">
            <select value={datasetKey} onChange={(event) => setDatasetKey(event.target.value)}>
              {datasets.map((dataset) => (
                <option key={dataset.key} value={dataset.key} disabled={!dataset.available}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Limit">
            <input type="number" min={1} max={500} value={limit} onChange={(event) => setLimit(event.target.value === "" ? "" : Number(event.target.value))} />
          </Field>
          <button className="primary-button w-full" onClick={runTesting} disabled={mutation.isPending || modelIds.length === 0 || !datasetKey}>
            <Play size={17} /> Run test
          </button>
          {mutation.error && <p className="error-text">{mutation.error.message}</p>}
        </section>
        <section className="panel">
          <HistoryHeader
            title="Testing Jobs"
            selectedCount={selectedIds.length}
            onRefresh={() => jobsQuery.refetch()}
            onDelete={confirmDeleteTestingRows}
            onClear={confirmClearTestingJobs}
            dataTour="testing-jobs"
          />
          <TestingJobTable
            jobs={jobsForTask}
            selectedIds={selectedIds}
            setSelectedIds={setSelectedIds}
            modelTaskById={modelTaskById}
            modelNameById={modelNameById}
            taskType={taskType}
          />
          {jobsQuery.isLoading && <TableSkeleton rows={5} />}
          <MutationError mutations={[deleteMutation]} />
        </section>
        <section className="panel workspace-grid-full">
          <PanelTitle icon={<BarChart3 size={18} />} title="Comparison" dataTour="testing-comparison" />
          <TestingComparison jobs={comparisonJobs} modelTaskById={modelTaskById} modelNameById={modelNameById} taskType={taskType} />
        </section>
      </div>
      {confirmationDialog}
    </div>
  );
}
