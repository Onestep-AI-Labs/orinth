"use client";

import { Activity, StopCircle } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { TrainingDetails } from "@/features/training/training-components";
import { activePollInterval, isActiveStatus } from "@/features/platform/utils";
import { Button, ButtonLink, MutationError, PageHeader, PageSkeleton, ProgressPanel, useConfirmationDialog } from "@/features/platform/ui";
import type { TrainingJob } from "@/types/api";

export function TrainingDetailPage({ jobId }: { jobId: string }) {
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const jobQuery = useQuery({
    queryKey: ["training-job", jobId],
    queryFn: () => api.trainingJob(jobId),
    refetchInterval: (query) => activePollInterval(query.state.data as TrainingJob | undefined)
  });
  const cancelMutation = useMutation({
    mutationFn: api.cancelTrainingJob,
    onSuccess: async () => {
      await jobQuery.refetch();
    }
  });
  const promoteMutation = useMutation({
    mutationFn: api.promoteTrainingJob,
    onSuccess: async () => {
      await jobQuery.refetch();
    }
  });
  const job = jobQuery.data;
  // The reverse link training-run → registered model: the model records its
  // originating job, so a completed run can jump straight to its artifact —
  // for LLM runs, that is where Save / Export (GGUF, merged, adapter) lives.
  const modelsQuery = useQuery({
    queryKey: ["models", "for-training-job", job?.project_id ?? ""],
    queryFn: () => api.models(false, job?.project_id),
    enabled: Boolean(job && job.status === "completed")
  });
  const registeredModel = modelsQuery.data?.find((model) => model.training_job_id === jobId);
  const isLlmRun = job?.model_family === "llm_sft" || (registeredModel?.family ?? "").startsWith("llm_");

  function confirmCancelTrainingJob(job: TrainingJob) {
    confirm({
      title: "Cancel training job?",
      message: `This will request cancellation for training job ${job.id.slice(0, 8)}. Partial artifacts may remain unavailable.`,
      confirmLabel: "Cancel job",
      tone: "warning",
      onConfirm: () => cancelMutation.mutate(job.id)
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Training Detail" subtitle={job?.id.slice(0, 8) ?? jobId.slice(0, 8)} icon={<Activity size={20} />} />
      {job ? (
        <section className="panel space-y-5">
          <div className="flex flex-wrap gap-2">
            {isActiveStatus(job.status) && (
              <Button variant="secondary" onClick={() => confirmCancelTrainingJob(job)}>
                <StopCircle size={16} /> Cancel
              </Button>
            )}
            {job.status === "completed" && Boolean(job.artifacts?.best_model) && !job.promoted_model_id && (
              <Button variant="secondary" onClick={() => promoteMutation.mutate(job.id)}>
                Promote
              </Button>
            )}
            {job.status === "completed" && (
              <ButtonLink variant="secondary" href="/testing">
                Test this model
              </ButtonLink>
            )}
            {job.status === "completed" && isLlmRun && registeredModel && (
              <ButtonLink href={`/models/${registeredModel.id}`}>
                Save / Export
              </ButtonLink>
            )}
          </div>
          {job.status === "completed" && isLlmRun && (
            <p className="training-llm-export-hint">
              Fine-tuning saved a LoRA adapter. Use Save / Export to convert it to GGUF (to serve and
              chat), merged 16-bit, or an adapter zip — no separate step needed.
            </p>
          )}
          <ProgressPanel progress={job.progress} status={job.status} error={job.error} hideLogs />
          <TrainingDetails job={job} />
          <MutationError mutations={[cancelMutation, promoteMutation]} />
        </section>
      ) : (
        <PageSkeleton title="Loading training job" />
      )}
      {confirmationDialog}
    </div>
  );
}

