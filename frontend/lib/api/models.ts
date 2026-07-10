import type { ModelInfo, TaskType } from "@/types/api";
import { jsonFetch, query } from "@/lib/api/client";

export const modelsApi = {
  models: (availableOnly = false, projectId?: string, taskType?: TaskType) =>
    jsonFetch<ModelInfo[]>(
      `/models${query({
        available_only: availableOnly ? "true" : undefined,
        project_id: projectId,
        task_type: taskType
      })}`
    )
};
