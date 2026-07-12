import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { jsonFetch, query } from "@/lib/api/client";

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init
  });
}

describe("jsonFetch", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    fetchMock.mockReset();
    vi.unstubAllGlobals();
  });

  it("resolves with the parsed JSON body on success", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: "abc", name: "Model A" }));

    const result = await jsonFetch<{ id: string; name: string }>("/models/abc");

    expect(result).toEqual({ id: "abc", name: "Model A" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/models/abc");
    expect(init?.headers).toMatchObject({ "Content-Type": "application/json" });
  });

  it("extracts the FastAPI detail field from an error response", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ detail: "Model not found" }), {
        status: 404,
        statusText: "Not Found",
        headers: { "Content-Type": "application/json" }
      })
    );

    await expect(jsonFetch("/models/missing")).rejects.toThrow("Model not found");
  });

  it("falls back to statusText when the error body is not JSON", async () => {
    fetchMock.mockResolvedValue(
      new Response("<html>Internal Server Error</html>", {
        status: 500,
        statusText: "Internal Server Error",
        headers: { "Content-Type": "text/html" }
      })
    );

    await expect(jsonFetch("/models")).rejects.toThrow("Internal Server Error");
  });

  it("does not force a Content-Type header when the body is FormData", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ ok: true }));

    const formData = new FormData();
    formData.set("file", "content");
    await jsonFetch("/upload", { method: "POST", body: formData });

    const [, init] = fetchMock.mock.calls[0];
    expect(init?.headers).not.toHaveProperty("Content-Type");
  });
});

describe("query", () => {
  it("builds a query string from defined, non-empty values", () => {
    expect(query({ page: 1, search: "granuloma" })).toBe("?page=1&search=granuloma");
  });

  it("omits undefined, null, and empty-string values", () => {
    expect(query({ page: 1, search: undefined, filter: null, name: "" })).toBe("?page=1");
  });

  it("returns an empty string when there are no usable params", () => {
    expect(query({ search: undefined, filter: null })).toBe("");
  });
});
