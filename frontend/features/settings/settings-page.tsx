"use client";

import { useEffect, useState } from "react";
import { KeyRound, Save, Settings, Sparkles, Trash2 } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useProject } from "@/components/app-shell";
import { api } from "@/lib/api";
import { Field, Metric, MutationError, PageHeader, PanelTitle, StatusBadge } from "@/features/platform/ui";

export function SettingsPage() {
  const { projects } = useProject();
  const [hfToken, setHfToken] = useState("");
  const [openrouterKey, setOpenrouterKey] = useState("");
  const [openrouterModel, setOpenrouterModel] = useState("");
  const settingsQuery = useQuery({ queryKey: ["platform-settings"], queryFn: api.platformSettings });
  const modelsQuery = useQuery({ queryKey: ["openrouter-models"], queryFn: api.openRouterModels });
  const updateSettings = useMutation({
    mutationFn: api.updatePlatformSettings,
    onSuccess: async () => {
      setHfToken("");
      setOpenrouterKey("");
      await settingsQuery.refetch();
    }
  });
  const tokenConfigured = Boolean(settingsQuery.data?.huggingface_hub_token_configured);
  const openrouterConfigured = Boolean(settingsQuery.data?.openrouter_api_key_configured);

  // Keep the model select in sync with the persisted default once it loads.
  useEffect(() => {
    setOpenrouterModel(settingsQuery.data?.openrouter_model ?? "");
  }, [settingsQuery.data?.openrouter_model]);

  function saveToken() {
    updateSettings.mutate({ huggingface_hub_token: hfToken.trim() || null });
  }

  function clearToken() {
    updateSettings.mutate({ huggingface_hub_token: null });
  }

  function saveOpenrouterKey() {
    updateSettings.mutate({ openrouter_api_key: openrouterKey.trim() || null });
  }

  function clearOpenrouterKey() {
    updateSettings.mutate({ openrouter_api_key: null });
  }

  function saveOpenrouterModel() {
    updateSettings.mutate({ openrouter_model: openrouterModel.trim() || null });
  }

  const openrouterModels = modelsQuery.data?.models ?? [];

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
        <PanelTitle icon={<KeyRound size={18} />} title="Hugging Face" dataTour="settings-hf" />
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
        <PanelTitle icon={<Sparkles size={18} />} title="OpenRouter" dataTour="settings-openrouter" />
        <p className="mt-1 text-sm text-ink-subtle">
          Powers LLM-assisted record generation in data recipes. The key is write-only — it is stored server-side
          and never returned to the browser. Without a key, recipes fall back to deterministic rule-based records.
        </p>
        <div className="metric-grid mt-4">
          <Metric label="API key" value={openrouterConfigured ? "Configured" : "Not configured"} />
          <Metric label="Default model" value={settingsQuery.data?.openrouter_model ?? "None"} />
        </div>
        <div className="settings-token-form mt-4">
          <Field label="OpenRouter API key">
            <input
              type="password"
              value={openrouterKey}
              onChange={(event) => setOpenrouterKey(event.target.value)}
              placeholder={openrouterConfigured ? "Key configured" : "Paste key"}
              autoComplete="off"
            />
          </Field>
          <div className="action-row">
            <button className="primary-button" type="button" onClick={saveOpenrouterKey} disabled={updateSettings.isPending}>
              <Save size={16} /> Save
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={clearOpenrouterKey}
              disabled={updateSettings.isPending || !openrouterConfigured}
            >
              <Trash2 size={16} /> Clear
            </button>
            <StatusBadge status={openrouterConfigured ? "available" : "missing"} />
          </div>
          <Field label="Default model">
            <select value={openrouterModel} onChange={(event) => setOpenrouterModel(event.target.value)}>
              <option value="">No default</option>
              {openrouterModels.map((model) => (
                <option key={model.id} value={model.id}>{model.name}</option>
              ))}
            </select>
          </Field>
          <div className="action-row">
            <button className="secondary-button" type="button" onClick={saveOpenrouterModel} disabled={updateSettings.isPending}>
              <Save size={16} /> Save model
            </button>
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
