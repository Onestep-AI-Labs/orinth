"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

/**
 * Hub browse data access.
 *
 * Three queries with three different lifetimes, which is why they are not one:
 * the vocabulary never changes within a session, a search changes on every
 * keystroke the user commits, and a preview is per dataset and expensive enough
 * that it must not refire when an unrelated filter moves.
 */

export type HubFilters = {
  query: string;
  modality: string[];
  format: string[];
  size: string[];
  task_category: string[];
  sort: string;
};

export const EMPTY_FILTERS: HubFilters = {
  query: "",
  modality: [],
  format: [],
  size: [],
  task_category: [],
  // Matches huggingface.co's own default, so an unfiltered browse here shows
  // the same datasets the website's front page does.
  sort: "trending"
};

/**
 * The filter vocabulary and its explanations.
 *
 * `staleTime: Infinity` because it is a constant compiled into the backend —
 * refetching it on window focus would be a request that can never return
 * anything new.
 */
export function useHubFacetsQuery() {
  return useQuery({
    queryKey: ["hub-facets"],
    queryFn: () => api.datasetHubFacets(),
    staleTime: Infinity,
    retry: false
  });
}

export function useHubSearchQuery(filters: HubFilters, limit = 30) {
  return useQuery({
    // Arrays are part of the key by value, so toggling a chip is a new query and
    // the previous result stays cached for when it is toggled back off.
    queryKey: ["hub-search", filters, limit],
    queryFn: () =>
      api.searchDatasetHub({
        query: filters.query || undefined,
        modality: filters.modality,
        format: filters.format,
        size: filters.size,
        task_category: filters.task_category,
        sort: filters.sort,
        limit
      }),
    // The Hub rate-limits; a failed search retried three times turns one 429
    // into four and makes the panel take four times as long to say so.
    retry: false,
    // Keeping the previous page mounted while the next loads is what stops the
    // grid collapsing to an empty state on every chip toggle.
    placeholderData: (previous) => previous
  });
}

export function useHubPreviewQuery(
  hubId: string | undefined,
  config: string | undefined,
  split: string | undefined,
  limit = 20
) {
  return useQuery({
    queryKey: ["hub-preview", hubId, config, split, limit],
    queryFn: () =>
      api.previewDatasetHub({ hub_id: hubId ?? "", config, split, limit }),
    enabled: Boolean(hubId),
    retry: false
  });
}
