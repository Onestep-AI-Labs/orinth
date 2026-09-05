"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { toast } from "@/features/platform/toast";
import type { DatasetHubIngestRequest, DatasetPrepPlan, DatasetPrepStatus } from "@/types/api";

/**
 * Prep-agent data access.
 *
 * Kept in its own module rather than added to `hooks.ts`: the prep surfaces own
 * their state (following `table-studio.tsx`), and `dataset-page.tsx` already
 * carries thirty-odd `useState` calls without this.
 */

/** Everything the agent touches, so one run refreshes the whole studio. */
function invalidateDataset(client: ReturnType<typeof useQueryClient>, datasetId: string) {
  return Promise.all([
    client.invalidateQueries({ queryKey: ["dataset-catalog"] }),
    client.invalidateQueries({ queryKey: ["dataset-items"] }),
    client.invalidateQueries({ queryKey: ["dataset-eda"] }),
    client.invalidateQueries({ queryKey: ["dataset-prep", datasetId] }),
    client.invalidateQueries({ queryKey: ["dataset-prep-status", datasetId] })
  ]);
}

/** States the agent passes through while a run is still going. */
const PREP_IN_FLIGHT = new Set(["detecting", "planning", "applying"]);

export function prepIsRunning(status: DatasetPrepStatus | null | undefined): boolean {
  return Boolean(status && PREP_IN_FLIGHT.has(status.state));
}

const POLL_INTERVAL_MS = 1500;

export function useDatasetPrepQuery(datasetId: string | undefined) {
  return useQuery({
    queryKey: ["dataset-prep", datasetId],
    // A dataset that has never met the agent 404s; that is an answer, not a
    // failure, so it must not retry or raise a toast.
    queryFn: () => api.datasetPrepPlan(datasetId ?? "").catch(() => null),
    enabled: Boolean(datasetId),
    retry: false
  });
}

/**
 * The run's own progress, polled while it is moving.
 *
 * Separate from `useDatasetPrepQuery` because it answers a different question at
 * a different cadence: the plan is what the agent *concluded* and is worth one
 * fetch, while this is what it is *doing* and is worth one every second and a
 * half — but only while something is happening. `/prep/status` is a single JSON
 * read for exactly this reason; polling `/datasets/{id}` instead would walk
 * every split on every tick, which on a large upload costs seconds and lands on
 * the same thread pool the run is using.
 *
 * `refetchInterval` returns false once the state settles, so a finished dataset
 * costs nothing.
 */
export function useDatasetPrepStatusQuery(datasetId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ["dataset-prep-status", datasetId],
    queryFn: () => api.datasetPrepStatus(datasetId ?? ""),
    enabled: Boolean(datasetId) && enabled,
    retry: false,
    refetchInterval: (query) =>
      prepIsRunning(query.state.data as DatasetPrepStatus | undefined) ? POLL_INTERVAL_MS : false
  });
}

/** Give up watching after this long. The run itself is unaffected; only this
 *  browser stops waiting, and the next catalog refresh still shows the truth. */
const POLL_TIMEOUT_MS = 15 * 60 * 1000;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Wait for a queued prep run to settle.
 *
 * `POST /prep` returns 202 as soon as the run is queued: applying a plan to a
 * large upload is tens of thousands of file operations, and holding the request
 * open for it is what the dev proxy used to reset. Progress is read from
 * `/prep/status`, which is a manifest read — the catalog costs seconds to build
 * on a large dataset, so polling *that* would recreate the load it replaced.
 */
async function waitForPrep(datasetId: string) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS);
    const status = await api.datasetPrepStatus(datasetId).catch(() => null);
    if (!status) continue;
    if (!PREP_IN_FLIGHT.has(status.state)) return status;
  }
  return null;
}

export function useRunPrepMutation(onDone?: () => void) {
  const client = useQueryClient();
  return useMutation({
    mutationKey: ["dataset-prep-run"],
    mutationFn: async ({
      datasetId,
      autoApply = true
    }: {
      datasetId: string;
      autoApply?: boolean;
    }) => {
      await api.runDatasetPrep(datasetId, { auto_apply: autoApply });
      // The mutation stays pending for the whole run, so every `isPending`
      // already wired to a disabled button keeps meaning "Orinth is working".
      const status = await waitForPrep(datasetId);
      return { datasetId, status };
    },
    onSuccess: async ({ datasetId, status }) => {
      await invalidateDataset(client, datasetId);
      if (status?.state === "failed") {
        toast.error(status.error ?? "Orinth could not prepare this dataset.");
        onDone?.();
        return;
      }
      const plan = await api.datasetPrepPlan(datasetId).catch(() => null);
      // The agent stopping is a normal outcome, not an error — it means the data
      // needs something only a person can give. It gets no toast either way: a
      // green banner reading "these images carry no labels" would misrepresent
      // it, and the Overview panel states the reason more durably.
      if (plan && !plan.needs_input) {
        toast.success(`Prepared as ${String(plan.task_type ?? "dataset").replace(/_/g, " ")}.`);
      }
      onDone?.();
    }
  });
}

export function useApplyPrepMutation(onDone?: () => void) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ datasetId, plan }: { datasetId: string; plan: DatasetPrepPlan }) =>
      api.applyDatasetPrep(datasetId, plan),
    onSuccess: async (result) => {
      await invalidateDataset(client, result.dataset.id);
      toast.success("Dataset prepared.");
      onDone?.();
    }
  });
}

export function useUndoPrepMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (datasetId: string) => api.undoDatasetPrep(datasetId),
    onSuccess: async (dataset) => {
      await invalidateDataset(client, dataset.id);
      toast.success("Reverted to the state before Orinth prepared this dataset.");
    }
  });
}

export function useDiscardStagedMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (datasetId: string) => api.discardStagedFiles(datasetId),
    onSuccess: async (dataset) => {
      await invalidateDataset(client, dataset.id);
      toast.success("Original upload discarded.");
    }
  });
}

/**
 * Upload staged files, reporting how much of the body has been sent.
 *
 * `onProgress` is threaded down to `XMLHttpRequest` (see `uploadFetch`): a
 * folder upload is minutes of a single request, and the only honest readout
 * during it is bytes sent. React Query has no notion of request progress, so
 * the caller owns the number and this only passes it through.
 */
export function useIngestDatasetMutation() {
  const client = useQueryClient();
  return useMutation({
    mutationKey: ["dataset-ingest"],
    mutationFn: ({
      form,
      onProgress
    }: {
      form: FormData;
      onProgress?: (fraction: number) => void;
    }) => api.ingestDataset(form, onProgress),
    onSuccess: async (dataset) => {
      await invalidateDataset(client, dataset.id);
    }
  });
}

/**
 * Import a HuggingFace split as it is: download it, and stop.
 *
 * Deliberately does *not* wait, and deliberately does not prepare. `POST
 * /import/hub/as-is` answers 202 with the draft and the download reports
 * through `/prep/status`, which the dataset's own Overview tab already polls
 * and renders — so the right move is to open the dataset, where the download is
 * a labelled progress bar and "Prepare with Orinth" is waiting underneath it.
 *
 * Preparing is a separate press because a Hub dataset is under no obligation to
 * be shaped like something Orinth trains. Chaining the two made a download that
 * worked perfectly report itself as a failure whenever the agent could not name
 * a task.
 */
export function useIngestHubDatasetMutation(onDone?: (datasetId: string) => void) {
  const client = useQueryClient();
  return useMutation({
    mutationKey: ["dataset-hub-ingest"],
    mutationFn: (payload: DatasetHubIngestRequest) => api.ingestDatasetHub(payload),
    onSuccess: async (dataset) => {
      await invalidateDataset(client, dataset.id);
      toast.success("Downloading from HuggingFace into your workspace.");
      onDone?.(dataset.id);
    },
    onError: (error: Error) => toast.error(error.message)
  });
}
