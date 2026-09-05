export const API_ORIGIN = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ?? "";
export const API_BASE = `${API_ORIGIN}/api`;

export function mediaUrl(path: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${API_ORIGIN}${path}`;
}

export function apiAssetUrl(path: string | null): string | null {
  return mediaUrl(path);
}

export async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
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
      // keep status text
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json() as Promise<T>;
}

/**
 * A POST whose upload progress is observable.
 *
 * `fetch` cannot report how much of a request body has been sent — there is no
 * upload-progress event and `ReadableStream` request bodies are not supported
 * where this has to run — so a multi-gigabyte folder upload showed the word
 * "Uploading" and nothing else for minutes. `XMLHttpRequest` is the one API in
 * the platform that does report it, which is why this is the exception to
 * `jsonFetch` rather than a preference.
 *
 * `onProgress` receives 0..1, and only while the browser knows the total; a
 * request whose length is not computable simply never calls it, and the caller
 * falls back to its indeterminate state.
 */
export function uploadFetch<T>(
  path: string,
  body: FormData,
  onProgress?: (fraction: number) => void
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE}${path}`);
    request.responseType = "text";

    if (onProgress) {
      request.upload.onprogress = (event) => {
        if (!event.lengthComputable || event.total <= 0) return;
        onProgress(Math.min(1, event.loaded / event.total));
      };
    }

    request.onerror = () => reject(new Error("The upload could not reach the backend."));
    request.onabort = () => reject(new Error("The upload was cancelled."));
    request.onload = () => {
      let payload: unknown = null;
      try {
        payload = JSON.parse(request.responseText);
      } catch {
        payload = null;
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(payload as T);
        return;
      }
      // Same error shape `jsonFetch` produces, so a caller cannot tell which
      // transport it used from the message it has to show.
      const detail = (payload as { detail?: unknown } | null)?.detail ?? request.statusText;
      reject(new Error(typeof detail === "string" ? detail : JSON.stringify(detail)));
    };

    request.send(body);
  });
}

export class ApiValidationError extends Error {
  constructor(path: string, issues: Array<{ path: PropertyKey[]; message: string }>) {
    const summary = issues
      .slice(0, 3)
      .map((issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`)
      .join("; ");
    super(`Unexpected response from ${path} — ${summary}`);
    this.name = "ApiValidationError";
  }
}

type SchemaLike = {
  safeParse: (data: unknown) =>
    | { success: true }
    | { success: false; error: { issues: Array<{ path: PropertyKey[]; message: string }> } };
};

export async function jsonFetchChecked<T>(path: string, schema: SchemaLike, init?: RequestInit): Promise<T> {
  const data = await jsonFetch<T>(path, init);
  const result = schema.safeParse(data);
  if (!result.success) {
    throw new ApiValidationError(path, result.error.issues);
  }
  return data;
}

export function query(
  params: Record<string, string | number | boolean | null | undefined | string[]>
): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    // An array becomes repeated keys (`?modality=text&modality=tabular`), which
    // is what FastAPI's `list[str] = Query(...)` reads and what the Hub's own
    // facet URLs look like — so a filter set in Orinth transfers to a
    // huggingface.co URL unchanged.
    if (Array.isArray(value)) {
      for (const entry of value) {
        if (entry !== "") search.append(key, entry);
      }
      continue;
    }
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  }
  const value = search.toString();
  return value ? `?${value}` : "";
}
