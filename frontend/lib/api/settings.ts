import type { components } from "@/types/generated/api";
import { jsonFetch } from "@/lib/api/client";

type PlatformSettings = components["schemas"]["PlatformSettingsRead"];
type PlatformSettingsUpdate = components["schemas"]["PlatformSettingsUpdate"];

export const settingsApi = {
  platformSettings: () => jsonFetch<PlatformSettings>("/settings"),
  updatePlatformSettings: (payload: PlatformSettingsUpdate) =>
    jsonFetch<PlatformSettings>("/settings", {
      method: "PATCH",
      body: JSON.stringify(payload)
    })
};
