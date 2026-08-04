import type { TaskType } from "@/types/api";
import type { components } from "@/types/generated/api";
import { API_BASE, jsonFetch, query } from "@/lib/api/client";

type Architecture = components["schemas"]["Architecture"];
type ArchitectureSummary = components["schemas"]["ArchitectureSummary"];
type ArchitectureGraph = components["schemas"]["ArchitectureGraph"];
type ArchitectureValidation = components["schemas"]["ArchitectureValidation"];
type ArchitectureTemplate = components["schemas"]["ArchitectureTemplate"];
type ArchitectureCode = components["schemas"]["ArchitectureCode"];
type ArchitectureCreate = components["schemas"]["ArchitectureCreate"];
type ArchitectureUpdate = components["schemas"]["ArchitectureUpdate"];
type NodeSpec = components["schemas"]["NodeSpec"];
type DeleteResponse = components["schemas"]["DeleteResponse"];

/** Which emitter renders the code panel. Keras is what trains in-platform. */
export type Framework = "keras" | "torch";

export const architecturesApi = {
  architectures: (projectId?: string, taskType?: TaskType) =>
    jsonFetch<ArchitectureSummary[]>(
      `/architectures${query({ project_id: projectId, task_type: taskType })}`
    ),
  architecture: (architectureId: string) =>
    jsonFetch<Architecture>(`/architectures/${encodeURIComponent(architectureId)}`),
  createArchitecture: (payload: Partial<ArchitectureCreate> & { name: string }) =>
    jsonFetch<Architecture>("/architectures", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  updateArchitecture: (architectureId: string, payload: Partial<ArchitectureUpdate>) =>
    jsonFetch<Architecture>(`/architectures/${encodeURIComponent(architectureId)}`, {
      method: "PUT",
      body: JSON.stringify(payload)
    }),
  deleteArchitecture: (architectureId: string) =>
    jsonFetch<DeleteResponse>(`/architectures/${encodeURIComponent(architectureId)}`, {
      method: "DELETE"
    }),
  duplicateArchitecture: (architectureId: string) =>
    jsonFetch<Architecture>(`/architectures/${encodeURIComponent(architectureId)}/duplicate`, {
      method: "POST"
    }),
  architectureNodeCatalog: (taskType?: TaskType) =>
    jsonFetch<NodeSpec[]>(`/architectures/node-catalog${query({ task_type: taskType })}`),
  architectureNodeCategories: () => jsonFetch<string[]>("/architectures/node-categories"),
  architectureTemplates: (taskType?: TaskType) =>
    jsonFetch<ArchitectureTemplate[]>(`/architectures/templates${query({ task_type: taskType })}`),
  // Stateless: validates the live canvas, so the studio never has to save a
  // half-finished graph just to see its shapes.
  validateArchitectureGraph: (graph: ArchitectureGraph, numClasses?: number) =>
    jsonFetch<ArchitectureValidation>(
      `/architectures/validate${query({ num_classes: numClasses?.toString() })}`,
      { method: "POST", body: JSON.stringify(graph) }
    ),
  architectureCode: (architectureId: string, numClasses = 2, framework: Framework = "keras") =>
    jsonFetch<ArchitectureCode>(
      `/architectures/${encodeURIComponent(architectureId)}/code${query({
        num_classes: numClasses.toString(),
        framework
      })}`
    ),
  architectureCodeDownloadUrl: (
    architectureId: string,
    numClasses = 2,
    framework: Framework = "keras"
  ) =>
    `${API_BASE}/architectures/${encodeURIComponent(architectureId)}/code/download` +
    `?num_classes=${numClasses}&framework=${framework}`,
  architectureExportUrl: (architectureId: string) =>
    `${API_BASE}/architectures/${encodeURIComponent(architectureId)}/export`,
  importArchitecture: (form: FormData) =>
    jsonFetch<Architecture>("/architectures/import", { method: "POST", body: form })
};
