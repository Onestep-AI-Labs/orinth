export { API_BASE, API_ORIGIN, apiAssetUrl, mediaUrl } from "@/lib/api/client";

import { architecturesApi } from "@/lib/api/architectures";
import { datasetsApi } from "@/lib/api/datasets";
import { inferenceApi } from "@/lib/api/inference";
import { modelsApi } from "@/lib/api/models";
import { projectsApi } from "@/lib/api/projects";
import { recipesApi } from "@/lib/api/recipes";
import { servingApi } from "@/lib/api/serving";
import { settingsApi } from "@/lib/api/settings";
import { testingApi } from "@/lib/api/testing";
import { trainingApi } from "@/lib/api/training";

export const api = {
  ...projectsApi,
  ...modelsApi,
  ...architecturesApi,
  ...datasetsApi,
  ...recipesApi,
  ...inferenceApi,
  ...servingApi,
  ...testingApi,
  ...trainingApi,
  ...settingsApi
};
