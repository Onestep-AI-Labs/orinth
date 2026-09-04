"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { toast } from "@/features/platform/toast";
import type { NotebookComputeTarget, NotebookRuntimeStatus } from "@/types/api";

/** Notebook list, templates, runtime, and runs. */

export function useNotebooksQuery(projectId: string) {
  return useQuery({
    queryKey: ["notebooks", projectId],
    queryFn: () => api.notebooks(projectId),
    enabled: Boolean(projectId)
  });
}

export function useNotebookQuery(notebookId: string | undefined) {
  return useQuery({
    queryKey: ["notebook", notebookId],
    queryFn: () => api.notebook(notebookId ?? ""),
    enabled: Boolean(notebookId)
  });
}

export function useNotebookTemplatesQuery() {
  return useQuery({
    queryKey: ["notebook-templates"],
    queryFn: () => api.notebookTemplates(),
    // Templates ship in the repo, so within a session they cannot change.
    staleTime: Infinity
  });
}

/**
 * Runtime status, polled only while it is moving.
 *
 * `starting` is the one state worth watching; once it settles the answer is
 * stable until someone presses a button, and polling a stopped runtime forever
 * is a request per second that can never say anything new.
 */
export function useNotebookRuntimeQuery() {
  return useQuery({
    queryKey: ["notebook-runtime"],
    queryFn: () => api.notebookRuntime(),
    refetchInterval: (query) =>
      (query.state.data as NotebookRuntimeStatus | undefined)?.state === "starting" ? 1000 : false
  });
}

//: What the picker offers before the probe has answered. `auto` is not a
//: placeholder for a real device — it *is* the default, and the one choice that
//: is correct on every machine — so the dropdown is usable from the first
//: paint instead of empty for seven seconds.
const AUTO_TARGET: NotebookComputeTarget[] = [
  {
    id: "auto",
    label: "Automatic",
    kind: "auto",
    available: true,
    detail: "Each framework uses whatever it can reach. Nothing is hidden from it."
  }
];

/**
 * What a kernel can run on. Probed hardware plus the declared remote providers.
 *
 * **The probe is slow the first time — measured at 7.1s cold on this machine**,
 * because it spawns a subprocess that imports torch and TensorFlow. That is the
 * right place for it (`docs/ai/rules.md` forbids importing either into the API),
 * but it meant a dropdown with nothing in it for seven seconds on a cold
 * backend. `placeholderData` gives it the one answer that is always true while
 * the rest arrives.
 *
 * `staleTime: Infinity` because a machine does not grow a GPU while the tab is
 * open, and the backend caches the probe for ten minutes anyway.
 */
export function useComputeTargetsQuery() {
  return useQuery({
    queryKey: ["notebook-compute-targets"],
    queryFn: () => api.notebookComputeTargets(),
    placeholderData: AUTO_TARGET,
    staleTime: Infinity
  });
}

export function useStartRuntimeMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationKey: ["notebook-runtime-start"],
    mutationFn: (device: string = "auto") => api.startNotebookRuntime(device),
    onSuccess: (status) => {
      client.setQueryData(["notebook-runtime"], status);
      if (status.state === "running") toast.success("Notebook runtime started.");
    },
    onError: (error: Error) => toast.error(error.message)
  });
}

export function useRestartRuntimeMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationKey: ["notebook-runtime-restart"],
    mutationFn: (device: string = "auto") => api.restartNotebookRuntime(device),
    onSuccess: (status) => {
      client.setQueryData(["notebook-runtime"], status);
      if (status.state === "running") toast.success(`Runtime restarted on ${status.device}.`);
    },
    onError: (error: Error) => toast.error(error.message)
  });
}

export function useStopRuntimeMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.stopNotebookRuntime(),
    onSuccess: (status) => client.setQueryData(["notebook-runtime"], status)
  });
}

export function useCreateNotebookMutation(onCreated: (id: string) => void) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: { project_id: string; name: string; template_id?: string | null }) =>
      api.createNotebook(payload),
    onSuccess: async (notebook) => {
      await client.invalidateQueries({ queryKey: ["notebooks"] });
      onCreated(notebook.id);
    },
    onError: (error: Error) => toast.error(error.message)
  });
}

export function useDuplicateNotebookMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (notebookId: string) => api.duplicateNotebook(notebookId),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["notebooks"] });
      toast.success("Notebook duplicated.");
    },
    onError: (error: Error) => toast.error(error.message)
  });
}

export function useDeleteNotebookMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (notebookId: string) => api.deleteNotebook(notebookId),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["notebooks"] });
      toast.success("Notebook deleted.");
    },
    onError: (error: Error) => toast.error(error.message)
  });
}

export function useRenameNotebookMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ notebookId, name }: { notebookId: string; name: string }) =>
      api.updateNotebook(notebookId, { name }),
    onSuccess: async (notebook) => {
      client.setQueryData(["notebook", notebook.id], notebook);
      await client.invalidateQueries({ queryKey: ["notebooks"] });
    },
    onError: (error: Error) => toast.error(error.message)
  });
}

export function useNotebookRunsQuery(notebookId: string | undefined) {
  return useQuery({
    queryKey: ["notebook-runs", notebookId],
    queryFn: () => api.notebookRuns(notebookId ?? ""),
    enabled: Boolean(notebookId),
    // A run is written by the kernel while a cell is still going, so the panel
    // has to re-read rather than wait for a signal it will never get.
    refetchInterval: 4000
  });
}

export function useNotebookRunQuery(notebookId: string | undefined, runId: string | undefined) {
  return useQuery({
    queryKey: ["notebook-run", notebookId, runId],
    queryFn: () => api.notebookRun(notebookId ?? "", runId ?? ""),
    enabled: Boolean(notebookId && runId),
    refetchInterval: 4000
  });
}
