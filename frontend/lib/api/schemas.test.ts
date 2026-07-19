import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiValidationError, jsonFetchChecked } from "./client";
import { jobProgressSchema, jobWithProgressSchema, projectListSchema } from "./schemas";

const validProgress = {
  current_item: null,
  current_step: "Queued",
  elapsed_seconds: 0,
  eta_seconds: null,
  finished_at: null,
  logs: [],
  percent: 0,
  processed: 0,
  started_at: null,
  total: null
};

describe("api schemas", () => {
  it("accepts a valid job progress payload", () => {
    expect(jobProgressSchema.safeParse(validProgress).success).toBe(true);
  });

  it("rejects corrupted job progress", () => {
    const result = jobProgressSchema.safeParse({ ...validProgress, percent: "90%" });
    expect(result.success).toBe(false);
  });

  it("accepts project summaries with unknown extra fields", () => {
    const result = projectListSchema.safeParse([
      {
        id: "p1",
        name: "Panoramic study",
        description: null,
        created_at: "2026-07-18T00:00:00Z",
        updated_at: "2026-07-18T00:00:00Z",
        metadata: { cohort: "A" },
        task_types: ["classification"]
      }
    ]);
    expect(result.success).toBe(true);
  });
});

describe("jsonFetchChecked", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubFetch(payload: unknown) {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(payload), { status: 200 }))
    );
  }

  it("returns data when the payload matches", async () => {
    const payload = { id: "job-1", status: "running", progress: validProgress, extra: true };
    stubFetch(payload);
    await expect(jsonFetchChecked("/training/jobs/job-1", jobWithProgressSchema)).resolves.toEqual(payload);
  });

  it("throws ApiValidationError naming the failing path on corrupt payloads", async () => {
    stubFetch({ id: "job-1", status: "running", progress: { ...validProgress, logs: "no" } });
    const promise = jsonFetchChecked("/training/jobs/job-1", jobWithProgressSchema);
    await expect(promise).rejects.toBeInstanceOf(ApiValidationError);
    await expect(
      jsonFetchChecked("/training/jobs/job-1", jobWithProgressSchema).catch((error: Error) => error.message)
    ).resolves.toContain("progress.logs");
  });
});
