import type { components } from "@/types/generated/api";
import { jsonFetch, query } from "@/lib/api/client";

type RecipeRead = components["schemas"]["RecipeRead"];
type RecipeCreate = components["schemas"]["RecipeCreate"];
type RecipeGenerateRequest = components["schemas"]["RecipeGenerateRequest"];
type RecipeRecord = components["schemas"]["RecipeRecord"];
type RecipeRecordPage = components["schemas"]["RecipeRecordPage"];
type RecipeCommitResponse = components["schemas"]["RecipeCommitResponse"];
type OpenRouterModelsResponse = components["schemas"]["OpenRouterModelsResponse"];
type DeleteResponse = components["schemas"]["DeleteResponse"];

export const recipesApi = {
  recipes: (projectId?: string) =>
    jsonFetch<RecipeRead[]>(`/recipes${query({ project_id: projectId })}`),
  recipe: (recipeId: string) => jsonFetch<RecipeRead>(`/recipes/${recipeId}`),
  createRecipe: (payload: RecipeCreate) =>
    jsonFetch<RecipeRead>("/recipes", { method: "POST", body: JSON.stringify(payload) }),
  deleteRecipe: (recipeId: string) =>
    jsonFetch<DeleteResponse>(`/recipes/${recipeId}`, { method: "DELETE" }),
  addRecipeSources: (recipeId: string, form: FormData) =>
    jsonFetch<RecipeRead>(`/recipes/${recipeId}/sources`, { method: "POST", body: form }),
  deleteRecipeSource: (recipeId: string, sourceId: string) =>
    jsonFetch<RecipeRead>(`/recipes/${recipeId}/sources/${sourceId}`, { method: "DELETE" }),
  generateRecipe: (recipeId: string, payload: RecipeGenerateRequest) =>
    jsonFetch<RecipeRead>(`/recipes/${recipeId}/generate`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  cancelRecipe: (recipeId: string) =>
    jsonFetch<RecipeRead>(`/recipes/${recipeId}/cancel`, { method: "POST" }),
  recipeRecords: (recipeId: string, params: { page?: number; page_size?: number }) =>
    jsonFetch<RecipeRecordPage>(
      `/recipes/${recipeId}/records${query({ page: params.page, page_size: params.page_size })}`
    ),
  updateRecipeRecord: (recipeId: string, index: number, record: Record<string, unknown>) =>
    jsonFetch<RecipeRecord>(`/recipes/${recipeId}/records/${index}`, {
      method: "PATCH",
      body: JSON.stringify({ record })
    }),
  addRecipeRecord: (recipeId: string, record: Record<string, unknown>) =>
    jsonFetch<RecipeRecord>(`/recipes/${recipeId}/records`, {
      method: "POST",
      body: JSON.stringify({ record })
    }),
  deleteRecipeRecords: (recipeId: string, indices: number[]) =>
    jsonFetch<RecipeRead>(`/recipes/${recipeId}/records/delete`, {
      method: "POST",
      body: JSON.stringify({ indices })
    }),
  commitRecipe: (recipeId: string, name?: string | null) =>
    jsonFetch<RecipeCommitResponse>(`/recipes/${recipeId}/commit`, {
      method: "POST",
      body: JSON.stringify({ name })
    }),
  openRouterModels: () => jsonFetch<OpenRouterModelsResponse>("/recipes/openrouter/models")
};
