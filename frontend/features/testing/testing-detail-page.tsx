"use client";

import { useMemo } from "react";
import { BarChart3, FlaskConical } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { MetricsDetails, TestingComparison } from "@/features/testing/testing-components";
import { TERMINAL_STATUSES } from "@/features/platform/constants";
import { activePollInterval } from "@/features/platform/utils";
import { PageHeader, PageSkeleton, PanelTitle, ProgressPanel, TableSkeleton } from "@/features/platform/ui";
import type { EvaluationJob, TaskType } from "@/types/api";

export function TestingDetailPage({ jobId }: { jobId: string }) {
  const { projectId } = useProject();
  const jobQuery = useQuery({
    queryKey: ["testing-job", jobId],
    queryFn: () => api.testingJob(jobId),
    refetchInterval: (query) => activePollInterval(query.state.data as EvaluationJob | undefined)
  });
  const modelsQuery = useQuery({
    queryKey: ["models", "available", projectId],
    queryFn: () => api.models(true, projectId)
  });
  const comparisonQuery = useQuery({
    queryKey: ["testing-comparison", jobId],
    queryFn: () => api.testingComparison(jobId),
    enabled: Boolean(jobQuery.data)
  });
  const perImageQuery = useQuery({
    queryKey: ["testing-per-image", jobId],
    queryFn: () => api.testingPerImage(jobId),
    enabled: Boolean(jobQuery.data && TERMINAL_STATUSES.has(jobQuery.data.status))
  });
  const job = jobQuery.data;
  const detailModels = modelsQuery.data ?? [];
  const modelTaskById = useMemo(
    () => Object.fromEntries(detailModels.map((model) => [model.id, model.task_type])) as Record<string, TaskType>,
    [detailModels]
  );
  const modelNameById = useMemo(
    () => Object.fromEntries(detailModels.map((model) => [model.id, model.name])) as Record<string, string>,
    [detailModels]
  );
  return (
    <div className="space-y-5">
      <PageHeader title="Testing Detail" subtitle={job?.id.slice(0, 8) ?? jobId.slice(0, 8)} icon={<FlaskConical size={20} />} />
      {job ? (
        <>
          <section className="panel">
            <PanelTitle icon={<BarChart3 size={18} />} title="Comparison" />
            {comparisonQuery.isLoading ? (
              <TableSkeleton rows={4} />
            ) : (
              <TestingComparison
                jobs={comparisonQuery.data?.jobs ?? [job]}
                modelTaskById={modelTaskById}
                modelNameById={modelNameById}
              />
            )}
          </section>
          <section className="panel space-y-5">
            <ProgressPanel progress={job.progress} status={job.status} error={job.error} />
            {perImageQuery.isLoading ? <TableSkeleton rows={6} /> : <MetricsDetails job={job} rows={perImageQuery.data ?? []} />}
          </section>
        </>
      ) : (
        <PageSkeleton title="Loading testing job" />
      )}
    </div>
  );
}

