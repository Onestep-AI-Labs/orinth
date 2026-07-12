import type { PlatformSettings, PlatformSettingsUpdate } from "@/types/api";
import { jsonFetch } from "@/lib/api/client";

export const settingsApi = {
  platformSettings: () => jsonFetch<PlatformSettings>("/settings"),
  updatePlatformSettings: (payload: PlatformSettingsUpdate) =>
    jsonFetch<PlatformSettings>("/settings", {
      method: "PATCH",
      body: JSON.stringify(payload)
    })
};
