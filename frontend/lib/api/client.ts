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

export function query(params: Record<string, string | number | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  }
  const value = search.toString();
  return value ? `?${value}` : "";
}
