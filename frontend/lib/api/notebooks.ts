import type { components } from "@/types/generated/api";
import { jsonFetch, query } from "@/lib/api/client";

type NotebookSummary = components["schemas"]["NotebookSummary"];
type NotebookTemplate = components["schemas"]["NotebookTemplate"];
type NotebookSession = components["schemas"]["NotebookSession"];
type NotebookRuntimeStatus = components["schemas"]["NotebookRuntimeStatus"];
type NotebookComputeTarget = components["schemas"]["NotebookComputeTarget"];
type NotebookRun = components["schemas"]["NotebookRun"];
type NotebookRunSeries = components["schemas"]["NotebookRunSeries"];
type DeleteResponse = components["schemas"]["DeleteResponse"];

/**
 * Notebook CRUD and runtime control.
 *
 * The `jupyter-server` proxy is deliberately absent from this module: the
 * kernel client talks to `/api/notebooks/proxy/*` through
 * `@jupyterlab/services`, which owns its own transport, and wrapping it here
 * would put a second HTTP layer in front of a protocol that already has one.
 */
export const notebooksApi = {
  notebooks: (projectId: string) =>
    jsonFetch<NotebookSummary[]>(`/notebooks${query({ project_id: projectId })}`),
  notebook: (notebookId: string) => jsonFetch<NotebookSummary>(`/notebooks/${notebookId}`),
  notebookTemplates: () => jsonFetch<NotebookTemplate[]>("/notebooks/templates"),
  createNotebook: (payload: { project_id: string; name: string; template_id?: string | null }) =>
    jsonFetch<NotebookSummary>("/notebooks", { method: "POST", body: JSON.stringify(payload) }),
  updateNotebook: (notebookId: string, payload: { name?: string; tags?: string[] }) =>
    jsonFetch<NotebookSummary>(`/notebooks/${notebookId}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  duplicateNotebook: (notebookId: string) =>
    jsonFetch<NotebookSummary>(`/notebooks/${notebookId}/duplicate`, { method: "POST" }),
  deleteNotebook: (notebookId: string) =>
    jsonFetch<DeleteResponse>(`/notebooks/${notebookId}`, { method: "DELETE" }),
  notebookSession: (notebookId: string) =>
    jsonFetch<NotebookSession>(`/notebooks/${notebookId}/session`),

  notebookRuntime: () => jsonFetch<NotebookRuntimeStatus>("/notebooks/runtime"),
  notebookComputeTargets: () =>
    jsonFetch<NotebookComputeTarget[]>("/notebooks/runtime/targets"),
  // `device` is fixed for the life of the server, because a kernel inherits its
  // environment at spawn — which is why it is a start argument and not a
  // setting that can be changed underneath a running kernel.
  startNotebookRuntime: (device = "auto") =>
    jsonFetch<NotebookRuntimeStatus>("/notebooks/runtime/start", {
      method: "POST",
      body: JSON.stringify({ device })
    }),
  // One call, not stop-then-start: the device only changes across a restart,
  // and two calls leave a gap another tab can start the old one in.
  restartNotebookRuntime: (device = "auto") =>
    jsonFetch<NotebookRuntimeStatus>("/notebooks/runtime/restart", {
      method: "POST",
      body: JSON.stringify({ device })
    }),
  stopNotebookRuntime: () =>
    jsonFetch<NotebookRuntimeStatus>("/notebooks/runtime/stop", { method: "POST" }),

  notebookRuns: (notebookId: string) =>
    jsonFetch<NotebookRun[]>(`/notebooks/${notebookId}/runs`),
  notebookRun: (notebookId: string, runId: string) =>
    jsonFetch<NotebookRunSeries>(`/notebooks/${notebookId}/runs/${runId}`)
};
