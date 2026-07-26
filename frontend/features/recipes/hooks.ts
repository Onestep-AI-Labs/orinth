"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { toast } from "@/features/platform/toast";
import type { RecipeCommitResponse, RecipeGenerateRequest, RecipeRead } from "@/types/api";

const TRANSIENT: RecipeRead["status"][] = ["extracting", "generating"];

export function useRecipesQuery(projectId: string) {
  return useQuery({
    queryKey: ["recipes", projectId],
    queryFn: () => api.recipes(projectId)
  });
}

export function useRecipeQuery(recipeId: string) {
  return useQuery({
    queryKey: ["recipe", recipeId],
    queryFn: () => api.recipe(recipeId),
    // Poll while a run is in flight; go quiet once it settles (platform
    // convention — status polling, no websockets).
    refetchInterval: (query) =>
      query.state.data && TRANSIENT.includes(query.state.data.status) ? 1500 : false
  });
}

export function useRecipeRecordsQuery(recipeId: string, page: number, enabled: boolean) {
  return useQuery({
    queryKey: ["recipe-records", recipeId, page],
    queryFn: () => api.recipeRecords(recipeId, { page, page_size: 50 }),
    enabled
  });
}

export function useOpenRouterModelsQuery() {
  return useQuery({ queryKey: ["openrouter-models"], queryFn: api.openRouterModels });
}

export function usePlatformSettingsQuery() {
  return useQuery({ queryKey: ["platform-settings"], queryFn: api.platformSettings });
}

export function useCreateRecipeMutation(projectId: string, onCreated: (recipe: RecipeRead) => void) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.createRecipe,
    onSuccess: async (recipe) => {
      await client.invalidateQueries({ queryKey: ["recipes", projectId] });
      onCreated(recipe);
    }
  });
}

export function useDeleteRecipeMutation(projectId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (recipeId: string) => api.deleteRecipe(recipeId),
    onSuccess: async () => {
      toast.success("Recipe deleted");
      await client.invalidateQueries({ queryKey: ["recipes", projectId] });
    }
  });
}

export function useAddSourcesMutation(recipeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (form: FormData) => api.addRecipeSources(recipeId, form),
    onSuccess: async (recipe) => {
      const warned = recipe.sources.filter((source) => source.warnings.length > 0).length;
      toast.success(warned ? `Sources added — ${warned} with warnings` : "Sources added");
      client.setQueryData(["recipe", recipeId], recipe);
    }
  });
}

export function useDeleteSourceMutation(recipeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (sourceId: string) => api.deleteRecipeSource(recipeId, sourceId),
    onSuccess: (recipe) => client.setQueryData(["recipe", recipeId], recipe)
  });
}

export function useGenerateRecipeMutation(recipeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: RecipeGenerateRequest) => api.generateRecipe(recipeId, payload),
    onSuccess: (recipe) => {
      toast.success("Generation started");
      client.setQueryData(["recipe", recipeId], recipe);
    }
  });
}

export function useCancelRecipeMutation(recipeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.cancelRecipe(recipeId),
    onSuccess: (recipe) => client.setQueryData(["recipe", recipeId], recipe)
  });
}

export function useUpdateRecordMutation(recipeId: string, page: number) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ index, record }: { index: number; record: Record<string, unknown> }) =>
      api.updateRecipeRecord(recipeId, index, record),
    onSuccess: async () => {
      toast.success("Record saved");
      await client.invalidateQueries({ queryKey: ["recipe-records", recipeId, page] });
    }
  });
}

export function useAddRecordMutation(recipeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (record: Record<string, unknown>) => api.addRecipeRecord(recipeId, record),
    onSuccess: async () => {
      toast.success("Record added");
      await client.invalidateQueries({ queryKey: ["recipe-records", recipeId] });
      await client.invalidateQueries({ queryKey: ["recipe", recipeId] });
    }
  });
}

export function useDeleteRecordsMutation(recipeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (indices: number[]) => api.deleteRecipeRecords(recipeId, indices),
    onSuccess: async (recipe) => {
      client.setQueryData(["recipe", recipeId], recipe);
      await client.invalidateQueries({ queryKey: ["recipe-records", recipeId] });
    }
  });
}

export function useCommitRecipeMutation(recipeId: string, onCommitted: (response: RecipeCommitResponse) => void) {
  return useMutation({
    mutationFn: (name?: string | null) => api.commitRecipe(recipeId, name),
    onSuccess: (response) => {
      toast.success(`Committed ${response.committed_records} records`, {
        action: { label: "Open dataset", href: `/datasets?dataset=${response.dataset.id}` }
      });
      onCommitted(response);
    }
  });
}
