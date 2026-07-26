import type { components } from "@/types/generated/api";
import { jsonFetch } from "@/lib/api/client";

type PlatformSettings = components["schemas"]["PlatformSettingsRead"];
// The backend only touches fields actually present in the request body (see
// the settings router), so callers send a partial patch — one secret at a time.
type PlatformSettingsUpdate = Partial<components["schemas"]["PlatformSettingsUpdate"]>;

export const settingsApi = {
  platformSettings: () => jsonFetch<PlatformSettings>("/settings"),
  updatePlatformSettings: (payload: PlatformSettingsUpdate) =>
    jsonFetch<PlatformSettings>("/settings", {
      method: "PATCH",
      body: JSON.stringify(payload)
    })
};
