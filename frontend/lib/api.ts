import type {
  EvaluationDataset,
  EvaluationJob,
  InferenceResult,
  ModelInfo,
  TrainingJob
} from "@/types/api";

export const API_ORIGIN = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const API_BASE = `${API_ORIGIN}/api`;

export function mediaUrl(path: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${API_ORIGIN}${path}`;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" })
    }
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // keep response status text
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json() as Promise<T>;
}

export const api = {
  models: () => jsonFetch<ModelInfo[]>("/models"),
  inferenceHistory: () => jsonFetch<InferenceResult[]>("/inference"),
  runInference: (form: FormData) =>
    jsonFetch<InferenceResult>("/inference", {
      method: "POST",
      body: form
    }),
  datasets: () => jsonFetch<EvaluationDataset[]>("/testing/datasets"),
  testingJobs: () => jsonFetch<EvaluationJob[]>("/testing/jobs"),
  createTestingJob: (payload: { model_id: string; dataset_key: string; limit?: number | null }) =>
    jsonFetch<EvaluationJob>("/testing/jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  trainingJobs: () => jsonFetch<TrainingJob[]>("/training/jobs"),
  createTrainingJob: (payload: {
    model_family: "yolo" | "unet_inception";
    epochs: number;
    image_size: number;
    batch_size: number;
  }) =>
    jsonFetch<TrainingJob>("/training/jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  promoteTrainingJob: (jobId: string) =>
    jsonFetch<{ model_id: string }>(`/training/jobs/${jobId}/promote`, {
      method: "POST"
    })
};
