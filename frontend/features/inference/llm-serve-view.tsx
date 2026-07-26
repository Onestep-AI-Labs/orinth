"use client";

import { useRouter } from "next/navigation";
import { MessagesSquare, Server, Square } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatSeconds } from "@/features/platform/utils";
import { Badge, Button, ButtonLink, Field, MutationError, PanelTitle, TaskSelect } from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import { ModelSourcePicker, type StartTarget } from "@/features/inference/model-source-picker";
import type { ModelInfo, ServingStatus, TaskType } from "@/types/api";

function servingTone(state: string): "ok" | "info" | "fail" | "neutral" {
  if (state === "running") return "ok";
  if (state === "starting" || state === "stopping") return "info";
  return "neutral";
}

/**
 * LLM branch of the inference page: a launcher that starts the managed
 * llama.cpp server for a GGUF model via the shared three-source picker
 * (Trained · Local · Hugging Face), then auto-redirects to the streaming chat.
 */
export function LlmServeView({
  taskType,
  taskOptions,
  onSelectTask,
  models,
  modelsLoading
}: {
  taskType: TaskType;
  taskOptions: TaskType[];
  onSelectTask: (task: TaskType) => void;
  models: ModelInfo[];
  selectedModel?: string;
  onSelectModel?: (modelId: string) => void;
  modelsLoading?: boolean;
}) {
  const queryClient = useQueryClient();
  const router = useRouter();

  const statusQuery = useQuery({
    queryKey: ["serving-status"],
    queryFn: () => api.servingStatus(),
    refetchInterval: (q) => {
      const state = (q.state.data as ServingStatus | undefined)?.state;
      return state === "starting" || state === "stopping" ? 1500 : 4000;
    }
  });
  const status = statusQuery.data;
  const state = status?.state ?? "stopped";
  const running = state === "running";

  const startServing = useMutation({
    mutationKey: ["serving-start"],
    mutationFn: (target: StartTarget) => api.startServing(target),
    onSuccess: async (next) => {
      toast.success(`Serving ${next.model_name ?? "model"}`);
      await queryClient.invalidateQueries({ queryKey: ["serving-status"] });
      router.push("/inference/chat");
    }
  });
  const stopServing = useMutation({
    mutationFn: () => api.stopServing(),
    onSuccess: async () => {
      toast.success("Serving stopped");
      await queryClient.invalidateQueries({ queryKey: ["serving-status"] });
    }
  });

  const busy =
    startServing.isPending || stopServing.isPending || state === "starting" || state === "stopping";

  return (
    <div className="workspace-grid">
      <section className="panel serve-panel">
        <PanelTitle icon={<Server size={18} />} title="Serve a model" />
        <Field label="Task">
          <TaskSelect value={taskType} onChange={onSelectTask} options={taskOptions} />
        </Field>

        <div className="serving-status-row">
          <Badge tone={running ? servingTone(state) : "neutral"}>{running ? state : state}</Badge>
          {running ? (
            <span className="serving-status-meta">
              {status?.model_name}
              {status?.port ? ` · port ${status.port}` : ""}
              {status?.uptime_seconds != null ? ` · up ${formatSeconds(status.uptime_seconds)}` : ""}
            </span>
          ) : (
            <span className="serving-status-meta">Pick a GGUF model to start serving.</span>
          )}
        </div>

        {running && (
          <div className="serving-actions">
            <ButtonLink variant="primary" href="/inference/chat">
              <MessagesSquare size={16} /> Open chat
            </ButtonLink>
            <Button variant="secondary" onClick={() => stopServing.mutate()} disabled={busy}>
              <Square size={16} /> Stop
            </Button>
          </div>
        )}

        <ModelSourcePicker
          registeredModels={models}
          onServe={(target) => startServing.mutate(target)}
          disabled={busy}
        />

        {state === "stopped" && status?.error ? (
          <div className="serving-error">
            <p className="error-text">{status.error}</p>
            {status.stderr_tail.length > 0 ? (
              <pre className="export-log-tail">{status.stderr_tail.slice(-8).join("\n")}</pre>
            ) : null}
          </div>
        ) : null}

        <MutationError mutations={[startServing, stopServing]} />
        {modelsLoading ? <p className="field-hint">Loading models…</p> : null}
      </section>
    </div>
  );
}
