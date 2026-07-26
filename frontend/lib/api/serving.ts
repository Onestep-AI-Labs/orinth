import type { components } from "@/types/generated/api";
import { API_BASE, jsonFetch, query } from "@/lib/api/client";

type ServingStatus = components["schemas"]["ServingStatus"];
type ChatRequest = components["schemas"]["ChatRequest"];
type ServingScanResult = components["schemas"]["ServingScanResult"];
type ServingBrowseResult = components["schemas"]["ServingBrowseResult"];
type ServingConfig = components["schemas"]["ServingConfig"];
type ServingPickResult = components["schemas"]["ServingPickResult"];
type HubModelSearchResponse = components["schemas"]["HubModelSearchResponse"];
type HubFilesResponse = components["schemas"]["HubFilesResponse"];
type HubDownloadResult = components["schemas"]["HubDownloadResult"];
type HubRecommendationsResponse = components["schemas"]["HubRecommendationsResponse"];
type WebSearchResponse = components["schemas"]["WebSearchResponse"];
export type WebSearchResult = components["schemas"]["WebSearchResult"];

// The backend applies auto defaults for the nullable fields, and takes either a
// registered model_id or a custom model_path (exactly one). openapi-typescript
// emits the nullable fields as required, so a hand-written shape is clearer here.
type ServingStartRequest = {
  model_id?: string;
  model_path?: string;
  context_length?: number | null;
  n_gpu_layers?: number | null;
};

/** One frame emitted by the chat SSE proxy (see backend serving router). */
export type ChatStreamEvent =
  | { type: "delta"; content: string }
  | {
      type: "usage";
      completion_tokens: number;
      prompt_tokens: number | null;
      tokens_per_second: number;
      time_to_first_token_seconds: number | null;
    }
  | { type: "error"; message: string };

export const servingApi = {
  servingStatus: () => jsonFetch<ServingStatus>("/serving/status"),
  startServing: (payload: ServingStartRequest) =>
    jsonFetch<ServingStatus>("/serving/start", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  stopServing: () => jsonFetch<ServingStatus>("/serving/stop", { method: "POST" }),
  scanServingModels: (path: string) =>
    jsonFetch<ServingScanResult>(`/serving/scan${query({ path })}`),
  browseServingModels: (path?: string) =>
    jsonFetch<ServingBrowseResult>(`/serving/browse${query({ path })}`),
  servingConfig: () => jsonFetch<ServingConfig>("/serving/config"),
  saveServingConfig: (modelsDir: string | null) =>
    jsonFetch<ServingConfig>("/serving/config", {
      method: "PUT",
      body: JSON.stringify({ models_dir: modelsDir })
    }),
  pickServingPath: (kind: "folder" | "file") =>
    jsonFetch<ServingPickResult>("/serving/pick", {
      method: "POST",
      body: JSON.stringify({ kind })
    }),
  searchHubModels: (queryText: string, format: "gguf" | "mlx") =>
    jsonFetch<HubModelSearchResponse>(`/serving/hf/search${query({ query: queryText, format })}`),
  recommendedHubModels: () =>
    jsonFetch<HubRecommendationsResponse>("/serving/hf/recommended"),
  hubModelFiles: (repoId: string, format: "gguf" | "mlx") =>
    jsonFetch<HubFilesResponse>(`/serving/hf/files${query({ repo_id: repoId, format })}`),
  downloadHubModel: (repoId: string, filename: string) =>
    jsonFetch<HubDownloadResult>("/serving/hf/download", {
      method: "POST",
      body: JSON.stringify({ repo_id: repoId, filename })
    }),
  webSearch: (queryText: string, maxResults = 5) =>
    jsonFetch<WebSearchResponse>("/serving/web-search", {
      method: "POST",
      body: JSON.stringify({ query: queryText, max_results: maxResults })
    }),

  /**
   * Stream a chat completion from the served model.
   *
   * SSE-over-fetch (not websockets) per the platform's no-websockets
   * convention. Passing an `AbortSignal` and aborting it cancels the request,
   * which the backend proxies to the llama.cpp server so generation stops
   * server-side too.
   */
  async *streamChat(
    payload: ChatRequest,
    signal?: AbortSignal
  ): AsyncGenerator<ChatStreamEvent, void, unknown> {
    const response = await fetch(`${API_BASE}/serving/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal
    });
    if (!response.ok || !response.body) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail ?? detail;
      } catch {
        // keep status text
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line.
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
        const dataLine = frame
          .split("\n")
          .find((line) => line.startsWith("data:"));
        if (!dataLine) continue;
        const data = dataLine.slice("data:".length).trim();
        if (!data || data === "[DONE]") continue;
        try {
          yield JSON.parse(data) as ChatStreamEvent;
        } catch {
          // Skip a malformed frame rather than tearing down the stream.
        }
      }
    }
  }
};
