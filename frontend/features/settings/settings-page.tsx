"use client";

import { useState } from "react";
import { KeyRound, Save, Settings, Trash2 } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useProject } from "@/components/app-shell";
import { api } from "@/lib/api";
import { Field, Metric, MutationError, PageHeader, PanelTitle, StatusBadge } from "@/features/platform/ui";

export function SettingsPage() {
  const { projects } = useProject();
  const [hfToken, setHfToken] = useState("");
  const settingsQuery = useQuery({ queryKey: ["platform-settings"], queryFn: api.platformSettings });
  const updateSettings = useMutation({
    mutationFn: api.updatePlatformSettings,
    onSuccess: async () => {
      setHfToken("");
      await settingsQuery.refetch();
    }
  });
  const tokenConfigured = Boolean(settingsQuery.data?.huggingface_hub_token_configured);

  function saveToken() {
    updateSettings.mutate({ huggingface_hub_token: hfToken.trim() || null });
  }

  function clearToken() {
    updateSettings.mutate({ huggingface_hub_token: null });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Settings" subtitle="Platform preferences" icon={<Settings size={20} />} />
      <section className="panel">
        <PanelTitle icon={<Settings size={18} />} title="Platform Overview" />
        <div className="metric-grid mt-4">
          <Metric label="Total projects" value={projects.length} />
          <Metric label="Account settings" value="Coming soon" />
          <Metric label="Usage limits" value="Local" />
          <Metric label="Storage mode" value="Workspace" />
        </div>
      </section>
      <section className="panel">
        <PanelTitle icon={<KeyRound size={18} />} title="Hugging Face" />
        <div className="metric-grid mt-4">
          <Metric label="Hub token" value={tokenConfigured ? "Configured" : "Not configured"} />
          <Metric label="Request mode" value={tokenConfigured ? "Authenticated" : "Unauthenticated"} />
        </div>
        <div className="settings-token-form mt-4">
          <Field label="HF token">
            <input
              type="password"
              value={hfToken}
              onChange={(event) => setHfToken(event.target.value)}
              placeholder={tokenConfigured ? "Token configured" : "Paste token"}
              autoComplete="off"
            />
          </Field>
          <div className="action-row">
            <button className="primary-button" type="button" onClick={saveToken} disabled={updateSettings.isPending}>
              <Save size={16} /> Save
            </button>
            <button className="secondary-button" type="button" onClick={clearToken} disabled={updateSettings.isPending || !tokenConfigured}>
              <Trash2 size={16} /> Clear
            </button>
            <StatusBadge status={tokenConfigured ? "available" : "missing"} />
          </div>
        </div>
        <MutationError mutations={[updateSettings]} />
      </section>
      <section className="panel">
        <PanelTitle icon={<Settings size={18} />} title="Usage Limits" />
        <div className="metric-grid mt-4">
          <Metric label="Compute policy" value="Local only" />
          <Metric label="Storage quota" value="Not enforced" />
          <Metric label="API keys" value={tokenConfigured ? "HF token set" : "Not configured"} />
        </div>
      </section>
    </div>
  );
}
