export { API_BASE, API_ORIGIN, apiAssetUrl, mediaUrl } from "@/lib/api/client";

import { datasetsApi } from "@/lib/api/datasets";
import { inferenceApi } from "@/lib/api/inference";
import { modelsApi } from "@/lib/api/models";
import { projectsApi } from "@/lib/api/projects";
import { testingApi } from "@/lib/api/testing";
import { trainingApi } from "@/lib/api/training";

export const api = {
  ...projectsApi,
  ...modelsApi,
  ...datasetsApi,
  ...inferenceApi,
  ...testingApi,
  ...trainingApi
};
