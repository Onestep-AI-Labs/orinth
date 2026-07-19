import { z } from "zod";
import type { components } from "@/types/generated/api";

type JobProgress = components["schemas"]["JobProgress"];
type ProjectSummary = components["schemas"]["ProjectSummary"];

/**
 * Runtime validation for the endpoints where corrupt data is dangerous:
 * the project list (drives the shell), job progress (drives polling and
 * termination), dataset items (drive the annotation editor), and
 * inference results. Schemas validate the fields the UI depends on and
 * pass unknown fields through; the `satisfies` tethers make drift against
 * the OpenAPI-generated types a typecheck failure.
 */

export const errorEnvelopeSchema = z.object({
  detail: z.union([z.string(), z.array(z.unknown()), z.record(z.string(), z.unknown())])
});

export const jobProgressSchema = z.object({
  current_item: z.string().nullable(),
  current_step: z.string(),
  elapsed_seconds: z.number(),
  eta_seconds: z.number().nullable(),
  finished_at: z.string().nullable(),
  logs: z.array(z.string()),
  percent: z.number(),
  processed: z.number(),
  started_at: z.string().nullable(),
  total: z.number().nullable()
}) satisfies z.ZodType<JobProgress>;

export const projectSummarySchema = z.looseObject({
  id: z.string(),
  name: z.string(),
  description: z.string().nullable(),
  created_at: z.string(),
  updated_at: z.string()
}) satisfies z.ZodType<Pick<ProjectSummary, "id" | "name" | "description" | "created_at" | "updated_at">>;

export const projectListSchema = z.array(projectSummarySchema);

export const jobWithProgressSchema = z.looseObject({
  id: z.string(),
  status: z.string(),
  progress: jobProgressSchema
});

export const jobListSchema = z.array(jobWithProgressSchema);

export const datasetItemPageSchema = z.looseObject({
  items: z.array(
    z.looseObject({
      id: z.string()
    })
  ),
  total: z.number(),
  limit: z.number(),
  offset: z.number()
});

export const inferenceResultListSchema = z.array(
  z.looseObject({
    id: z.string(),
    created_at: z.string()
  })
);
