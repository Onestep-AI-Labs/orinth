"use client";

import { Activity, StopCircle } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { TrainingDetails } from "@/features/training/training-components";
import { activePollInterval, isActiveStatus } from "@/features/platform/utils";
import { MutationError, PageHeader, PageSkeleton, ProgressPanel, useConfirmationDialog } from "@/features/platform/ui";
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
              <button className="secondary-button" onClick={() => confirmCancelTrainingJob(job)}>
                <StopCircle size={16} /> Cancel
              </button>
            )}
            {job.status === "completed" && Boolean(job.artifacts?.best_model) && !job.promoted_model_id && (
              <button className="secondary-button" onClick={() => promoteMutation.mutate(job.id)}>
                Promote
              </button>
            )}
          </div>
          <ProgressPanel progress={job.progress} status={job.status} error={job.error} />
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

